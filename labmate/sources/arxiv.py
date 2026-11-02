"""arXiv: full text of a paper from its HTML version.

The arXiv API returns metadata only, so full text comes from arxiv.org/html/<id>,
which exists for many (not all) papers; it is generated from the paper's LaTeX.
Nothing here calls a language model.

arXiv's terms ask for no more than one request every three seconds, one
connection at a time, counted across everything you run. Every request goes
through one lock and a minimum gap, so parallel tool calls queue up instead of
breaking that. Parsed papers are kept in memory for the session only; the terms
don't allow storing and serving e-prints, so nothing is written to disk.
"""

import re
import threading
import time

import requests

from labmate import paper_text

BASE_URL = "https://arxiv.org"
MIN_INTERVAL = 3.0  # seconds between requests
_HEADERS = {"User-Agent": "labmate/0.1 (personal research tool)"}

_lock = threading.Lock()
_last_request = 0.0
_documents = {}  # arXiv id -> Document, this session only

_ID = r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})"
_ID_PATTERNS = [
    re.compile(r"arxiv\.org/(?:abs|pdf|html)/" + _ID, re.I),
    re.compile(r"^\s*arxiv:\s*" + _ID, re.I),
    re.compile(r"10\.48550/arxiv\." + _ID, re.I),
    re.compile(r"^\s*" + _ID + r"(?:v\d+)?\s*$"),
]


class ArxivError(Exception):
    pass


class ArxivUnavailable(Exception):
    """arXiv has no readable HTML version of this paper."""


def extract_id(text):
    """The arXiv ID in text (an ID, arXiv:ID, an arxiv.org URL, or an arXiv
    DOI), without the version, or None."""
    for pattern in _ID_PATTERNS:
        m = pattern.search(text or "")
        if m:
            return re.sub(r"(\.pdf)$", "", m.group(1))
    return None


def id_from_work(work):
    """The arXiv ID of an OpenAlex work (raw JSON with `doi` and `locations`), or None."""
    found = extract_id(work.get("doi") or "")
    if found:
        return found
    for loc in work.get("locations") or []:
        for key in ("landing_page_url", "pdf_url"):
            url = loc.get(key) or ""
            if "arxiv.org" in url:
                found = extract_id(url)
                if found:
                    return found
    return None


def _fetch(url):
    global _last_request
    with _lock:  # one request at a time, spaced out
        wait = _last_request + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            response = requests.get(url, headers=_HEADERS, timeout=30)
        except requests.RequestException as exc:
            raise ArxivError(f"can't reach arXiv: {exc}") from exc
        finally:
            _last_request = time.monotonic()
    return response


def get_document(arxiv_id):
    """The paper's Document. Raises ArxivUnavailable if arXiv has no HTML
    version, ArxivError on a network or server problem."""
    if arxiv_id in _documents:
        return _documents[arxiv_id]
    response = _fetch(f"{BASE_URL}/html/{arxiv_id}")
    if response.status_code == 404 or "/html/" not in response.url:
        raise ArxivUnavailable("arXiv has no HTML version of this paper")
    if not response.ok:
        raise ArxivError(f"arXiv returned {response.status_code}")
    document = paper_text.parse_ltx_html(response.text)
    if document is None:
        raise ArxivUnavailable("the HTML version has no readable text (its conversion from LaTeX may have failed)")
    _documents[arxiv_id] = document
    return document
