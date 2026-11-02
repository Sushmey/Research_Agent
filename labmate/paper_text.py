"""A paper's full text as sections, and how to find and page through them.

Source-agnostic: a source (arxiv.py today) turns its format into a Document,
and the read_paper tool works on that. parse_ltx_html() handles arXiv's HTML
versions, which are generated from LaTeX (LaTeXML) and mark sections as
<section class="ltx_section">, <section class="ltx_subsection">, and so on.

Tables and displayed equations are left out (they are noisy as plain text);
captions, paragraphs, lists and inline math (as its LaTeX) are kept. The
bibliography is dropped.
"""

import re
from html.parser import HTMLParser

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_SKIP_TAGS = {"script", "style", "nav", "svg", "table"}
_TEXT_TAGS = {"p", "li", "figcaption"}
_BREAK_TAGS = {"p", "li", "figcaption"}
_LEVELS = {
    "ltx_section": 1,
    "ltx_appendix": 1,
    "ltx_subsection": 2,
    "ltx_subsubsection": 3,
    "ltx_paragraph": 4,
    "ltx_subparagraph": 5,
}

# What people ask for, and the words that appear in section titles.
_ALIASES = {
    "conclusion": ["conclusion", "concluding", "closing remarks"],
    "future": ["future work", "future direction", "future research", "future scope", "future", "outlook", "open problem", "open question"],
    "limitation": ["limitation", "threats to validity", "shortcoming"],
    "abstract": ["abstract"],
    "introduction": ["introduction"],
    "related": ["related work", "background", "prior work"],
    "method": ["method", "approach", "framework", "model"],
    "experiment": ["experiment", "evaluation", "result"],
    "discussion": ["discussion"],
}


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections = []  # every section, in document order
        self._open = []  # sections currently open, innermost last
        self._stack = []  # (tag, kind) for every open element
        self._skip = 0  # inside something to ignore
        self._in_text = 0  # inside a paragraph, list item or caption
        self._title_for = None  # section whose heading is being read
        self.doc_title = ""
        self._in_doc_title = False

    def _new_section(self, sec_id, level, title="", appendix=False):
        sec = {"id": sec_id, "level": level, "title": title, "appendix": appendix, "parts": [], "titled": bool(title)}
        self.sections.append(sec)
        self._open.append(sec)
        return sec

    def _emit(self, text):
        if self._skip:
            return
        if self._in_doc_title:
            self.doc_title += text
            return
        if not self._open:
            return
        if self._title_for is not None:
            self._title_for["title"] += text
        elif self._in_text:
            self._open[-1]["parts"].append(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if tag in _VOID:
            if tag == "br":
                self._emit(" ")
            return
        kind = None
        level = next((_LEVELS[c] for c in classes if c in _LEVELS), None)
        if tag == "section" and level:
            self._new_section(attrs.get("id", ""), level, appendix="ltx_appendix" in classes)
            kind = "section"
        elif tag == "div" and "ltx_abstract" in classes:
            self._new_section("abstract", 1, title="Abstract")
            kind = "section"
        elif tag in _SKIP_TAGS or classes & {"ltx_bibliography", "ltx_ERROR"}:
            self._skip += 1
            kind = "skip"
        elif tag == "math":
            alt = attrs.get("alttext")
            if alt:
                self._emit(alt)
            self._skip += 1
            kind = "skip"
        elif tag == "h1" and "ltx_title_document" in classes and not self._open:
            self._in_doc_title = True
            kind = "doctitle"
        elif re.fullmatch(r"h[1-6]", tag) and "ltx_title" in classes and self._open and not self._open[-1]["titled"]:
            self._title_for = self._open[-1]
            kind = "title"
        elif tag in _TEXT_TAGS:
            self._in_text += 1
            kind = "text"
        self._stack.append((tag, kind))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag in _VOID or not any(t == tag for t, _ in self._stack):
            return
        while self._stack:
            open_tag, kind = self._stack.pop()
            if kind == "section":
                self._open.pop()
            elif kind == "skip":
                self._skip -= 1
            elif kind == "title":
                self._title_for["titled"] = True
                self._title_for = None
            elif kind == "doctitle":
                self._in_doc_title = False
            elif kind == "text":
                self._in_text -= 1
                if open_tag in _BREAK_TAGS and self._open and not self._skip:
                    self._open[-1]["parts"].append("\n\n")
            if open_tag == tag:
                break

    def handle_data(self, data):
        self._emit(data)


def _tidy(parts):
    text = "".join(parts)
    paragraphs = (" ".join(p.split()) for p in text.split("\n\n"))
    return "\n\n".join(p for p in paragraphs if p)


class Document:
    """A paper as a flat list of sections in reading order. Each section's own
    text excludes its subsections; text(i) includes them."""

    def __init__(self, sections, title=""):
        self.sections = sections
        self.title = title  # the paper's own title, as the source prints it

    def _end(self, i):
        j = i + 1
        while j < len(self.sections) and self.sections[j]["level"] > self.sections[i]["level"]:
            j += 1
        return j

    def text(self, i):
        sec = self.sections[i]
        out = [sec["text"]] if sec["text"] else []
        for k in range(i + 1, self._end(i)):
            child = self.sections[k]
            if child["text"]:
                out.append(f"{child['title'] or child['id']}\n{child['text']}")
        return "\n\n".join(out)

    def outline(self, max_level=2, limit=40):
        rows = []
        for i, sec in enumerate(self.sections):
            if sec["level"] <= max_level and (sec["title"] or sec["text"]):
                rows.append({"section": sec["title"] or sec["id"], "chars": len(self.text(i))})
        return rows[:limit]

    def find(self, query):
        """Indexes of sections matching a request like "conclusion", "future
        work", "6" or a word from a title, best first: main sections before
        subsections, body before appendices, then document order."""
        q = " ".join(query.lower().split())
        if not q:
            return []
        if re.fullmatch(r"[\d.]+", q):
            hits = [i for i, s in enumerate(self.sections) if s["title"].lower().startswith(q + " ") or s["title"].lower() == q]
        else:
            terms = []
            for key, words in _ALIASES.items():
                if key in q:
                    terms += words
            terms = terms or [q]
            hits = [i for i, s in enumerate(self.sections) if any(t in s["title"].lower() for t in terms)]
        return sorted(hits, key=lambda i: (self.sections[i]["appendix"], self.sections[i]["level"], i))


def parse_ltx_html(html):
    """A Document from an arXiv HTML page, or None if no sections were found."""
    parser = _Parser()
    parser.feed(html)
    parser.close()
    sections = []
    for s in parser.sections:
        s["title"] = " ".join(s["title"].split())
        sections.append(
            {"id": s["id"], "level": s["level"], "title": s["title"], "appendix": s["appendix"], "text": _tidy(s["parts"])}
        )
    if not any(s["text"] for s in sections):
        return None
    return Document(sections, title=" ".join(parser.doc_title.split()))


def page(text, offset=0, max_chars=2000):
    """(chunk, next_offset): up to max_chars of text from offset, ending at a
    paragraph or sentence boundary when one is near. next_offset is None at the end."""
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        offset = 0
    end = offset + max_chars
    if end >= len(text):
        return text[offset:], None
    window = text[offset:end]
    floor = int(len(window) * 0.6)
    for marker in ("\n\n", ". "):
        cut = window.rfind(marker)
        if cut >= floor:
            end = offset + cut + len(marker)
            break
    return text[offset:end].rstrip(), end
