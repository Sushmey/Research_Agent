"""Terminal output. Everything except the assistant's answer goes to stderr, so
`labmate ask "..." > answer.txt` captures only the answer.

    progress()  what the agent is doing (tool calls, its notes); hidden by --quiet
    warn()      problems worth seeing even with --quiet
"""

import sys

_quiet = False


def set_quiet(value):
    global _quiet
    _quiet = bool(value)


def _dim(text):
    # Dimmed on a real terminal, plain when piped.
    return f"\033[2m{text}\033[0m" if sys.stderr.isatty() else text


def progress(message):
    if not _quiet:
        print(_dim(message), file=sys.stderr, flush=True)


def warn(message):
    print(message, file=sys.stderr, flush=True)


def _markdown(text):
    """Rich's Markdown, with tables drawn as a full grid: a border, a line
    between rows and space inside the cells. Rich's own tables have only a line
    under the header, so rows with wrapped text run together. This relies on
    Rich's TableElement internals; if they change, plain Markdown is used."""
    from rich.markdown import Markdown

    try:
        from rich import box
        from rich.markdown import TableElement
        from rich.table import Table

        class GridTable(TableElement):
            def __rich_console__(self, console, options):
                table = Table(
                    box=box.SQUARE,
                    show_lines=True,
                    pad_edge=True,
                    style="markdown.table.border",
                )
                if self.header is not None and self.header.row is not None:
                    for column in self.header.row.cells:
                        heading = column.content.copy()
                        heading.stylize("markdown.table.header")
                        table.add_column(heading)
                if self.body is not None:
                    for row in self.body.rows:
                        table.add_row(*[cell.content for cell in row.cells])
                yield table

        class GridMarkdown(Markdown):
            elements = {**Markdown.elements, "table_open": GridTable}

        return GridMarkdown(text)
    except (ImportError, AttributeError):
        return Markdown(text)


def render(text, file=None):
    """Print the assistant's answer. On a terminal the markdown is rendered
    (tables, bold, lists) with Rich; when output is piped or redirected, or
    Rich isn't installed or fails, the raw markdown is printed so files and
    scripts get clean text."""
    file = file or sys.stdout
    if file.isatty():
        try:
            from rich.console import Console

            Console(file=file).print(_markdown(text))
            return
        except Exception:  # never lose an answer to a display problem
            pass
    print(text, file=file, flush=True)


def ask(prompt):
    """Prompt on stderr and read a line from stdin, so a prompt never lands in
    redirected output. Raises EOFError / KeyboardInterrupt like input()."""
    print(prompt, end="", file=sys.stderr, flush=True)
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\n")
