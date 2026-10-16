"""search_papers: keyword search over OpenAlex."""

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
                "year_from": {"type": "integer", "description": "Only papers published in this year or later."},
                "year_to": {"type": "integer", "description": "Only papers published in this year or earlier."},
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


def prompt_hint(options):
    if "max_calls" in options:
        return (
            f"Use search_papers to look for the idea. You get at most {options['max_calls']} "
            "searches per question, so make each a differently-worded query (the "
            "method alone, the problem alone, synonyms)."
        )
    return (
        "Use search_papers to look for the idea. Search several differently-worded "
        "queries (the method alone, the problem alone, synonyms) before concluding anything."
    )


def run(query, year_from=None, year_to=None, limit=None, sort="relevance", options=None):
    opts = {**_DEFAULTS, **(options or {})}
    cap = opts["limit"]
    try:
        wanted = cap if limit is None else int(limit)
    except (TypeError, ValueError):
        wanted = cap
    return openalex.search_works(
        query,
        year_from,
        year_to,
        limit=max(1, min(wanted, cap)),
        sort=sort,
        abstract_chars=opts["abstract_chars"],
    )
