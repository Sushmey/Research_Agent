"""search_papers: keyword search over OpenAlex."""

import datetime

from labmate.sources import openalex

SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": (
            "Keyword search over papers (title and abstract) in OpenAlex. Use it "
            "to check whether an idea or method already exists and to find "
            "related work. Returns papers with title, authors, publication "
            "date, citation count, the start of the abstract and a retraction "
            "status. Try several differently-worded queries; one phrasing "
            "can miss papers another finds. Each search has a small cost, so "
            "don't repeat the same query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing the idea, e.g. 'contrastive learning for protein folding'.",
                },
                "year_from": {
                    "type": "integer",
                    "description": "Only papers published in this year or later. Leave out unless the user gave a period; never pass 0.",
                },
                "year_to": {
                    "type": "integer",
                    "description": "Only papers published in this year or earlier. Leave out unless the user gave a period.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results wanted. The system may cap it.",
                },
                "sort": {
                    "type": "string",
                    "enum": ["relevance", "cited", "recent"],
                    "description": "relevance (default), cited (most cited first), or recent (newest first).",
                },
            },
            "required": ["query"],
        },
    },
}

# Settings a profile may give this tool.
#   limit           most results one search returns
#   abstract_chars  how much of each abstract a search result includes
OPTIONS = {"limit": (int, 1, 25), "abstract_chars": (int, 100, 4000)}
_DEFAULTS = {"limit": 8, "abstract_chars": 1000}


_NO_YEARS = (
    " Don't set year_from or year_to unless the user gave a time period (in their message "
    "or in an answer to your question); they hide older papers that may be the original work."
)


def prompt_hint(options):
    if "max_calls" in options:
        return (
            f"Use search_papers to look for the idea. You get at most {options['max_calls']} "
            "searches per question, so make each a differently-worded query (the "
            "method alone, the problem alone, synonyms)." + _NO_YEARS
        )
    return (
        "Use search_papers to look for the idea. Search several differently-worded "
        "queries (the method alone, the problem alone, synonyms) before concluding anything."
        + _NO_YEARS
    )


def _year(value):
    """A usable publication year, or None. Models often pass 0 or null to mean
    "no filter", or a year that can't be real."""
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1000 <= year <= datetime.date.today().year + 1 else None


def clean_arguments(arguments):
    """The model's arguments with unusable year filters removed, so a 0 is
    neither applied nor shown as if it were a real filter."""
    cleaned = dict(arguments)
    for key in ("year_from", "year_to"):
        if key in cleaned:
            year = _year(cleaned[key])
            if year is None:
                del cleaned[key]
            else:
                cleaned[key] = year
    return cleaned


def run(query, year_from=None, year_to=None, limit=None, sort="relevance", options=None):
    opts = {**_DEFAULTS, **(options or {})}
    cap = opts["limit"]
    try:
        wanted = cap if limit is None else int(limit)
    except (TypeError, ValueError):
        wanted = cap
    return openalex.search_works(
        query,
        _year(year_from),
        _year(year_to),
        limit=max(1, min(wanted, cap)),
        sort=sort,
        abstract_chars=opts["abstract_chars"],
    )
