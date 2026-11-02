"""read_paper: read parts of a paper's full text (outline, or one section)."""

import re

from labmate import paper_text
from labmate.sources import arxiv, openalex

SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_paper",
        "description": (
            "Read part of a paper's full text. With only `id` it returns the outline "
            "(section titles and sizes). With `section` (a name such as 'conclusion', "
            "'future work', 'limitations', 'introduction', 'method', or a section "
            "number) it returns that section's text; `offset` continues a long "
            "section where the last call stopped. Authors say what is still open or "
            "unsolved in their conclusion and future work, which makes those sections "
            "the best place to check whether an idea has been done. Full text exists "
            "only for some papers (mostly arXiv); otherwise it says so and you should "
            "use the abstract from get_paper. Tables and displayed equations are left out."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "OpenAlex ID, DOI, or arXiv ID / URL of the paper.",
                },
                "section": {
                    "type": "string",
                    "description": "Section to read, e.g. 'conclusion' or 'future work'. Omit for the outline.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character position to continue from. Use next_offset from the previous result.",
                },
            },
            "required": ["id"],
        },
    },
}

# Settings a profile may give this tool.
#   max_chars  most text returned per call (keeps results inside a small context window)
OPTIONS = {"max_chars": (int, 500, 20000)}
_DEFAULTS = {"max_chars": 3000}


def prompt_hint(options):
    return (
        "Use read_paper to read a paper's full text in parts: call it with just the id "
        "to see the outline, then ask for a section by name. To judge whether an idea is "
        "still open, read the 'conclusion' or 'future work' of the closest papers and see "
        "what the authors say is unsolved. Only some papers have full text; if it is "
        "unavailable, rely on the abstract from get_paper and say so."
    )


def _words(title):
    return set(re.findall(r"[a-z0-9]+", (title or "").lower()))


def _same_paper(title_a, title_b):
    """Whether two titles name the same paper: most of the shorter title's words
    appear in the other. If either title is missing, they can't be matched."""
    a, b = _words(title_a), _words(title_b)
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= 0.6


def _unavailable(record, reason):
    return {
        "id": record["id"] if record else None,
        "title": record["title"] if record else None,
        "status": record["status"] if record else "unverified",
        "full_text": "unavailable",
        "reason": reason + ". Only the abstract is available (get_paper).",
    }


def run(id, section=None, offset=0, options=None):
    opts = {**_DEFAULTS, **(options or {})}

    # Find the paper, its retraction status and its arXiv version.
    arxiv_id = arxiv.extract_id(id)
    record = raw = None
    try:
        lookup_id = f"doi:10.48550/arXiv.{arxiv_id}" if arxiv_id else id
        record, raw = openalex.lookup(lookup_id)
    except openalex.OpenAlexError:
        if not arxiv_id:
            raise  # the model gave an ID OpenAlex doesn't know
    if raw and not arxiv_id:
        arxiv_id = arxiv.id_from_work(raw)

    # The retraction gate: a retracted paper is never fetched.
    if record and record["status"] == "RETRACTED":
        return {
            "id": record["id"],
            "title": record["title"],
            "status": "RETRACTED",
            "note": "Retracted per the Retraction Watch database. Its text was not fetched. "
            "Do NOT recommend this paper or rely on its claims.",
        }
    if not arxiv_id:
        return _unavailable(record, "No arXiv version was found for this paper")

    try:
        document = arxiv.get_document(arxiv_id)
    except arxiv.ArxivUnavailable as exc:
        reason = str(exc)
        return _unavailable(record, reason[:1].upper() + reason[1:])

    head = {
        "id": record["id"] if record else None,
        "arxiv_id": arxiv_id,
        "title": record["title"] if record else document.title or None,
        "published": record["published"] if record else None,
        "age": record.get("age") if record else None,
        # Without an OpenAlex record the retraction status could not be checked.
        "status": record["status"] if record else "unverified",
        "source": "arXiv HTML",
    }
    # OpenAlex sometimes attaches an arXiv page to the wrong work, so the
    # status above may belong to a different paper. Trust it only if the titles agree.
    if record and not _same_paper(record["title"], document.title):
        head.update(
            id=None,
            published=None,  # the record's date belongs to the other paper too
            age=None,
            title=document.title or None,
            status="unverified",
            status_note=f"OpenAlex's record for this ID is a different paper ({record['title']!r}), "
            "so the retraction status could not be verified for this one.",
        )
    if not section:
        return {
            **head,
            "outline": document.outline(),
            "next": "Call read_paper again with section set to one of these (or a name like 'conclusion').",
        }

    hits = document.find(section)
    if not hits:
        return {**head, "error": f"No section matches {section!r}.", "outline": document.outline()}
    text = document.text(hits[0])
    try:
        start = max(0, int(offset))
    except (TypeError, ValueError):
        start = 0
    chunk, next_offset = paper_text.page(text, start, opts["max_chars"])
    result = {
        **head,
        "section": document.sections[hits[0]]["title"],
        "text": chunk,
        "offset": start,
        "total_chars": len(text),
        "next_offset": next_offset,
    }
    others = [document.sections[i]["title"] for i in hits[1:4]]
    if others:
        result["other_matches"] = others
    return result
