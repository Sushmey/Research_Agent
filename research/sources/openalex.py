"""OpenAlex client: search and look up papers, with the retraction check built
in. Nothing here calls a language model.

Costs (from OpenAlex's docs): a keyword search is $0.001, a filtered list
$0.0001, a single-work lookup is free. The daily budget is $0.10 without an
API key and $1.00 with a free one, so every response is cached and spend is
tracked. The key is read from OPENALEX_API_KEY (via config.py) and is sent as
a Bearer header, so it never appears in a URL or a log line.

Every record returned goes through _to_record(), which is the integrity gate:
a work that OpenAlex flags is_retracted (Retraction Watch data) comes back as
a stub with no abstract, so the model can see that it exists and is retracted
but has nothing to build on.
"""

import time

import requests

from research import config

BASE_URL = "https://api.openalex.org"
SEARCH_COST = 0.001
LIST_COST = 0.0001

# Only these fields are requested, which keeps responses small for models
# with short context windows.
_SELECT = ",".join(
    [
        "id",
        "doi",
        "title",
        "publication_date",
        "publication_year",
        "cited_by_count",
        "is_retracted",
        "abstract_inverted_index",
        "authorships",
        "primary_location",
        "type",
    ]
)

_cache = {}
_spent = 0.0

# Every record that has been shown to the model this session, by OpenAlex ID.
# A final answer can be checked against this so it never cites a paper that
# wasn't actually retrieved (and gated).
SEEN = {}


class OpenAlexError(Exception):
    pass


def spent():
    return _spent


def _headers():
    key = config.get("OPENALEX_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _get(path, params, cost):
    """GET with caching, spend tracking and backoff on 429 / 5xx."""
    global _spent
    cache_key = (path, tuple(sorted(params.items())))
    if cache_key in _cache:
        return _cache[cache_key]

    last = None
    for attempt in range(3):
        try:
            response = requests.get(
                f"{BASE_URL}{path}", params=params, headers=_headers(), timeout=30
            )
        except requests.RequestException as exc:
            last = f"network error: {exc}"
            time.sleep(2**attempt)
            continue
        if response.status_code == 403 and "key_not_activated" in response.text:
            raise OpenAlexError(
                "OPENALEX_API_KEY is set but not activated yet. Click the "
                "activation link OpenAlex emailed you, then try again."
            )
        if response.status_code in (429, 500, 502, 503, 504):
            last = f"{response.status_code} from OpenAlex"
            time.sleep(2**attempt)
            continue
        if not response.ok:
            raise OpenAlexError(f"{response.status_code} from OpenAlex: {response.text[:200]}")
        _spent += cost
        data = response.json()
        _cache[cache_key] = data
        return data
    raise OpenAlexError(f"OpenAlex is not responding ({last}).")


def _abstract(inverted_index):
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
    if not inverted_index:
        return None
    words = {}
    for word, positions in inverted_index.items():
        for p in positions:
            words[p] = word
    return " ".join(words[i] for i in sorted(words))


def _to_record(work, abstract_chars=1500):
    """One OpenAlex work -> the compact, gated record the model sees."""
    work_id = (work.get("id") or "").replace("https://openalex.org/", "")
    doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
    authors = [
        a["author"]["display_name"]
        for a in (work.get("authorships") or [])[:4]
        if a.get("author")
    ]
    source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")

    record = {
        "id": work_id,
        "title": work.get("title"),
        "authors": authors,
        "published": work.get("publication_date"),
        "venue": source,
        "doi": doi,
        "cited_by_count": work.get("cited_by_count"),
    }

    if work.get("is_retracted"):
        record["status"] = "RETRACTED"
        record["note"] = (
            "Retracted per the Retraction Watch database. Do NOT recommend this "
            "paper or rely on its claims. Abstract withheld."
        )
    else:
        # "not_flagged" is deliberately not "clean": it means only that this
        # source has no retraction record, not that the paper is verified.
        record["status"] = "not_flagged" if work.get("is_retracted") is False else "unverified"
        text = _abstract(work.get("abstract_inverted_index"))
        if text and len(text) > abstract_chars:
            text = text[:abstract_chars].rstrip() + " ..."
        record["abstract"] = text or "(no abstract available)"

    SEEN[work_id] = {k: record[k] for k in ("id", "title", "published", "status")}
    return record


def search_works(query, year_from=None, year_to=None, limit=8, sort="relevance", abstract_chars=1500):
    """Keyword search over title and abstract. Costs $0.001."""
    params = {
        "search": query,
        "per_page": max(1, min(int(limit), 25)),
        "select": _SELECT,
    }
    if sort == "cited":
        params["sort"] = "cited_by_count:desc"
    elif sort == "recent":
        params["sort"] = "publication_date:desc"

    filters = ["type:!paratext"]
    if year_from and year_to:
        filters.append(f"publication_year:{int(year_from)}-{int(year_to)}")
    elif year_from:
        filters.append(f"publication_year:>{int(year_from) - 1}")
    elif year_to:
        filters.append(f"publication_year:<{int(year_to) + 1}")
    params["filter"] = ",".join(filters)

    data = _get("/works", params, SEARCH_COST)
    return {
        "total_matches": data.get("meta", {}).get("count"),
        "results": [_to_record(w, abstract_chars) for w in data.get("results", [])],
    }


def get_work(identifier):
    """Look up one paper by OpenAlex ID (W123...) or DOI. Free."""
    ident = identifier.strip()
    if ident.lower().startswith("10."):
        ident = f"doi:{ident}"
    elif ident.lower().startswith("https://doi.org/"):
        ident = f"doi:{ident}"
    elif ident.lower().startswith("https://openalex.org/"):
        ident = ident.rsplit("/", 1)[-1]
    data = _get(f"/works/{ident}", {"select": _SELECT}, 0.0)
    return _to_record(data, abstract_chars=4000)
