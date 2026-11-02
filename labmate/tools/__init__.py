"""The tools the model can call. Each tool is one file with:

    SCHEMA        what the model sees
    run(...)      what executes; takes the model's arguments plus options=,
                  the profile's settings for this tool (never set by the model)
    OPTIONS       {name: (type, min, max)} for the settings a profile may give
    prompt_hint   optional function(options) -> one line for the system prompt

To add a tool, write the file and add it to _MODULES below. Which tools the
model actually gets is decided by the profile (labmate/profiles.py).
"""

import json

from labmate.sources import arxiv, openalex
from labmate.tools import ask_user, get_paper, read_paper, search_papers, suggest_queries

_MODULES = {
    m.SCHEMA["function"]["name"]: m
    for m in (ask_user, suggest_queries, search_papers, get_paper, read_paper)
}

NAMES = set(_MODULES)

# The tools that return papers, and so apply the retraction check. Without
# them the assistant has no way to look anything up, so no profile may drop them.
REQUIRED = {"search_papers", "get_paper"}

# Options every tool accepts. max_calls is enforced by the agent loop.
_COMMON_OPTIONS = {"max_calls": (int, 1, 50)}

OPTION_SPECS = {
    name: {**_COMMON_OPTIONS, **getattr(m, "OPTIONS", {})} for name, m in _MODULES.items()
}


def _offered(name, module, profile, interactive):
    """Whether the model gets this tool: enabled by the profile, and a tool that
    needs the user is offered only when someone is there to answer."""
    return name in profile.tools and (interactive or not getattr(module, "NEEDS_USER", False))


def schemas(profile, interactive=False):
    """Tool definitions for the tools this profile enables."""
    return [m.SCHEMA for name, m in _MODULES.items() if _offered(name, m, profile, interactive)]


def prompt_hints(profile, interactive=False):
    """System-prompt lines contributed by the enabled tools."""
    hints = []
    for name, m in _MODULES.items():
        if _offered(name, m, profile, interactive) and hasattr(m, "prompt_hint"):
            hints.append(m.prompt_hint(profile.tools[name]))
    return hints


def clean_arguments(name, arguments):
    """The arguments as they will actually be run. A tool may define
    clean_arguments() to drop or fix values the model gets wrong (like a year
    of 0), so what is shown to the user matches what runs."""
    if not isinstance(arguments, dict):
        return {}
    fix = getattr(_MODULES.get(name), "clean_arguments", None)
    return fix(arguments) if fix else dict(arguments)


def run_tool(name, arguments, profile):
    """Run one tool call and return its result as a JSON string. Errors come
    back as {"error": ...} so the model can adjust instead of the loop dying."""
    if name not in profile.tools:
        return json.dumps(
            {"error": f"tool {name!r} is not available. Available: {', '.join(profile.tools)}"}
        )
    if not isinstance(arguments, dict):
        arguments = {}
    # The model must not be able to set the profile's options itself.
    arguments = {k: v for k, v in clean_arguments(name, arguments).items() if k != "options"}
    try:
        result = _MODULES[name].run(**arguments, options=profile.tools[name])
        return json.dumps(result, ensure_ascii=False)
    except TypeError as exc:  # wrong or missing argument names
        return json.dumps({"error": f"bad arguments for {name}: {exc}"})
    except (openalex.OpenAlexError, arxiv.ArxivError) as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:  # keep the loop alive on anything unexpected
        return json.dumps({"error": f"{name} failed: {exc}"})
