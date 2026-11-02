"""A table of the papers an answer mentions, built from what the tools actually
returned, never from the model's text. It shows when each paper is from and its
retraction status, and it exposes any paper the answer cites that was never
retrieved (an invented or misremembered reference).

Papers are found in the answer by OpenAlex ID (W123...) and by exact title.
"""

import re

from labmate import dates
from labmate.sources import openalex

_ID = re.compile(r"\bW\d{5,}\b")
# Models often write typographic dashes and quotes in titles; OpenAlex uses plain ones.
_PLAIN = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"} | {c: "'" for c in "\u2018\u2019\u201b"} | {c: '"' for c in "\u201c\u201d"})


def _norm(text):
    """Lowercase, plain dashes and quotes, single spaces: for comparing titles."""
    return " ".join(str(text or "").translate(_PLAIN).lower().split())
_STATUS = {
    "RETRACTED": "**RETRACTED**",
    "not_flagged": "no retraction found",
    "unverified": "unverified",
}


def _cell(text, limit=70):
    text = " ".join(str(text or "").split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def sources_table(answer, seen=None):
    """A markdown table of the papers named in `answer`, in order of first
    mention, or "" if it names none. `seen` is the papers retrieved this run."""
    seen = openalex.SEEN if seen is None else seen
    text = answer or ""
    lowered = _norm(text)

    found = {}  # id -> (position in the answer, record or None)
    for match in _ID.finditer(text):
        found.setdefault(match.group(0), (match.start(), seen.get(match.group(0))))
    for paper_id, record in seen.items():
        title = _norm(record.get("title"))
        if paper_id not in found and len(title) >= 12 and title in lowered:
            found[paper_id] = (lowered.index(title), record)
    if not found:
        return ""

    rows = ["| Paper | OpenAlex ID | Published | Age | Retraction status |", "|---|---|---|---|---|"]
    for paper_id, (_, record) in sorted(found.items(), key=lambda item: item[1][0]):
        if record is None:
            rows.append(f"| *not retrieved in this session* | {paper_id} | | | check before relying on it |")
            continue
        published = record.get("published")
        rows.append(
            "| {} | {} | {} | {} | {} |".format(
                _cell(record.get("title")),
                paper_id,
                dates.display(published) or "unknown",
                dates.age(published) or "",
                _STATUS.get(record.get("status"), record.get("status") or "unverified"),
            )
        )
    return "\n\n**Papers mentioned**\n\n" + "\n".join(rows)
