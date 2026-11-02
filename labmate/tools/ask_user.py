"""ask_user: ask the person a short clarifying question.

Unlike the other tools this one needs the user's terminal, so the agent loop
handles it directly (see agent._ask_user) instead of calling run(). It is only
offered to the model when someone is there to answer (an interactive terminal).
"""

import re

NEEDS_USER = True  # the registry hides this tool when there is nobody to ask

MAX_CHOICES = 6

SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Ask the user one short question and wait for the answer. Use it when their "
            "message is ambiguous in a way that would change what you search: a key term "
            "with more than one meaning in research, or a vague time period like 'recent' "
            "or 'latest'. Offer 2-4 choices so they can answer with a number. Don't use it "
            "for minor details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "One short, specific question."},
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 likely answers, each written out in full (not just 1, 2, 3). Don't repeat them in the question. The user can also type their own.",
                },
            },
            "required": ["question"],
        },
    },
}


_MARK = r"(?:\d{1,2}[).]|[A-Da-d]\))"  # "1)", "2.", "a)"
_BARE_LABEL = re.compile(rf"^\(?(?:\d{{1,2}}|[A-Da-d])[).:]?$")
_LEADING_MARK = re.compile(rf"^\(?{_MARK}?\s*")


def normalize(question, choices):
    """(question, choices) as they should be shown. Models sometimes write the
    options into the question and pass only labels ("1", "2", "3") as choices, or
    number the choices themselves; either would print as a confusing double list."""
    question = " ".join(str(question or "").split())
    choices = [" ".join(str(c).split()) for c in (choices or [])]
    choices = [c for c in choices if c]

    if choices and all(_BARE_LABEL.match(c) for c in choices):
        # Take the real options out of the question text, if they are there.
        inline = re.search(rf"\s{_MARK}\s", question)
        parts = []
        if inline:
            parts = [p.strip() for p in re.split(rf"\s*{_MARK}\s+", question[inline.start():]) if p.strip()]
        if len(parts) >= 2:
            question, choices = question[: inline.start()].strip(), parts
        else:
            choices = []  # labels alone mean nothing; let the user answer in words
    else:
        choices = [_LEADING_MARK.sub("", c, count=1) or c for c in choices]
    return question, choices[:MAX_CHOICES]


def prompt_hint(options):
    return (
        "Use ask_user to ask the user one short question, with 2-4 choices, when their "
        "message is ambiguous in a way that would change what you search: a key term with "
        "more than one meaning (for example 'harness' could be an LLM agent harness or a "
        "test harness), or a vague period like 'recent'. Don't ask about minor details; "
        "make a sensible assumption and say what you assumed."
    )


def run(question=None, choices=None, options=None):
    # Only reached if the model calls it when nobody can answer.
    return {"error": "No user is available to answer. Make a reasonable assumption and say what you assumed."}
