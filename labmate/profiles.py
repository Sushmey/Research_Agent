"""Profiles: which tools the model gets, and how each is set up.

A small local model needs more scaffolding than a large hosted one (a query
planner, tight result limits, short abstracts), and a large model should not
pay for scaffolding it doesn't need. A profile is a table of tools:

    "tool_name": False          hidden from the model
    "tool_name": True           enabled, with the tool's own defaults
    "tool_name": {"limit": 3}   enabled, with these options

LLM_PROFILE in .env picks one:
    auto (default)  small for a local model (Ollama, or an OpenAI-compatible
                    server on localhost), large for hosted services
    small | large   the built-in profiles below
    <name>          a profile defined in profiles.toml at the repo root

profiles.toml can define new profiles, replace a built-in one, or tweak one
with `extends` (see profiles.example.toml). Tools not mentioned are inherited
from the parent profile, or off if there is none. Tools in REQUIRED (the ones
that apply the retraction check) can't be turned off in any profile.

This module doesn't import the tools; the caller passes in what exists, so
there's no circular import.
"""

import os
import tomllib
from types import SimpleNamespace
from urllib.parse import urlparse

from labmate import config

PROFILES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles.toml"
)

BUILTIN = {
    "small": {
        "tools": {
            "ask_user": {"max_calls": 2},
            "suggest_queries": True,
            "search_papers": {"limit": 3, "abstract_chars": 500, "max_calls": 3},
            "get_paper": True,
            "read_paper": {"max_chars": 2000, "max_calls": 4},
        }
    },
    "large": {
        "tools": {
            "ask_user": {"max_calls": 2},
            "suggest_queries": False,
            "search_papers": {"limit": 8, "abstract_chars": 1500, "max_calls": 8},
            "get_paper": True,
            "read_paper": {"max_chars": 6000, "max_calls": 10},
        }
    },
}

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
_SPEC_KEYS = {"extends", "tools"}


class ProfileError(Exception):
    pass


def _is_local():
    """(is_local, why) for the configured model service."""
    provider = config.get("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return True, "ollama runs locally"
    if provider == "openai_compatible":
        host = urlparse(config.get("LLM_BASE_URL", "")).hostname
        if host in _LOCAL_HOSTS:
            return True, f"{provider} at {host}"
    return False, f"hosted provider {provider}"


def _load_user_profiles(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ProfileError(f"Can't read {os.path.basename(path)}: {exc}") from exc
    return {str(k).lower(): v for k, v in data.items()}


def _check_spec(name, spec, where):
    if not isinstance(spec, dict):
        raise ProfileError(f"Profile {name!r} ({where}) must be a table.")
    extra = set(spec) - _SPEC_KEYS
    if extra:
        raise ProfileError(
            f"Profile {name!r} ({where}): unknown key(s) {sorted(extra)}. Allowed: extends, tools."
        )
    if "extends" in spec and not isinstance(spec["extends"], str):
        raise ProfileError(f"Profile {name!r} ({where}): extends must be a profile name.")
    tools = spec.get("tools", {})
    if not isinstance(tools, dict):
        raise ProfileError(f"Profile {name!r} ({where}): tools must be a table.")
    for tool, value in tools.items():
        if not isinstance(value, (bool, dict)):
            raise ProfileError(
                f"Profile {name!r} ({where}): tool {tool!r} must be true, false, or a table of options."
            )


def _flatten(name, defs, stack, where):
    """Apply `extends` and return {tool: False | {options}} for one profile."""
    if name in stack:
        raise ProfileError(f"Profile {name!r} extends itself through {' -> '.join(stack + (name,))}.")
    spec = defs[name]
    _check_spec(name, spec, where(name))

    parent = spec.get("extends")
    if parent is None:
        merged = {}
    elif parent == name:
        # A profile that replaces a built-in one and extends it by the same name.
        if defs is BUILTIN or name not in BUILTIN:
            raise ProfileError(f"Profile {name!r} extends itself, but there is no built-in {name!r}.")
        # The built-in table is separate and has no `extends`, so it can't loop;
        # the same name appearing on the stack here is not a cycle.
        merged = _flatten(name, BUILTIN, (), lambda n: "built-in")
    elif parent in defs:
        merged = _flatten(parent, defs, stack + (name,), where)
    else:
        raise ProfileError(
            f"Profile {name!r} ({where(name)}) extends unknown profile {parent!r}. "
            f"Available: {', '.join(sorted(defs))}."
        )

    merged = dict(merged)
    for tool, value in spec.get("tools", {}).items():
        current = merged.get(tool, False)
        inherited = current if current is not False else {}
        if value is False:
            merged[tool] = False
        elif value is True:
            merged[tool] = inherited  # enabled, keeping any inherited options
        else:
            merged[tool] = {**inherited, **value}
    return merged


def _check_option(profile, tool, key, value, spec):
    kind, lo, hi = spec
    if isinstance(value, bool) or not isinstance(value, kind) or not lo <= value <= hi:
        raise ProfileError(
            f"Profile {profile!r}: option {key!r} of tool {tool!r} must be a whole "
            f"number from {lo} to {hi}, got {value!r}."
        )


def resolve(known, required, option_specs, name=None, user_file=PROFILES_FILE):
    """The profile in effect, as SimpleNamespace(name, tools, reason).

    known         names of the tools that exist
    required      tools that may not be turned off
    option_specs  {tool: {option: (type, min, max)}}
    `tools` in the result maps each enabled tool to its options dict.
    """
    user = _load_user_profiles(user_file)
    defs = {**BUILTIN, **user}
    file_label = os.path.basename(user_file)

    def where(n):
        return f"from {file_label}" if n in user else "built-in"

    requested = (name or config.get("LLM_PROFILE", "auto")).strip().lower()
    if requested == "auto":
        local, why = _is_local()
        chosen = "small" if local else "large"
        reason = f"auto: {why}"
    else:
        chosen = requested
        reason = "set by --profile" if name else "set by LLM_PROFILE"
    if chosen not in defs:
        raise ProfileError(
            f"LLM_PROFILE={chosen!r} is not a known profile. Available: {', '.join(sorted(defs))}."
        )

    merged = _flatten(chosen, defs, (), where)
    parent = defs[chosen].get("extends")
    if parent:
        reason += f", extends {parent}"
    if chosen in user:
        reason += f", from {file_label}"

    for tool in merged:
        if tool not in known:
            raise ProfileError(
                f"Profile {chosen!r} ({where(chosen)}): unknown tool {tool!r}. "
                f"Known tools: {', '.join(sorted(known))}."
            )
    for tool in required:
        if merged.get(tool) is False:
            raise ProfileError(
                f"Profile {chosen!r} ({where(chosen)}): {tool!r} can't be turned off; "
                "it applies the retraction check."
            )
        merged.setdefault(tool, {})

    tools = {}
    for tool, options in merged.items():
        if options is False:
            continue
        specs = option_specs.get(tool, {})
        for key, value in options.items():
            if key not in specs:
                raise ProfileError(
                    f"Profile {chosen!r} ({where(chosen)}): tool {tool!r} has no option {key!r}. "
                    f"Options: {', '.join(sorted(specs)) or 'none'}."
                )
            _check_option(chosen, tool, key, value, specs[key])
        tools[tool] = options
    return SimpleNamespace(name=chosen, tools=tools, reason=reason)


def describe(profile):
    return f"{profile.name} ({profile.reason})"
