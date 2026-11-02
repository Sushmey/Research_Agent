"""The labmate command line.

    labmate                      interactive session
    labmate ask "your question"  one question; only the answer goes to stdout
    labmate profile              which model and tools are in effect
    labmate history [-n N]       the recent conversation
    labmate history --clear      make the assistant forget it (the log file keeps it)

    --profile NAME   use a profile for this run instead of LLM_PROFILE
    -q, --quiet      hide the progress lines (tool calls and the model's notes)

Press Ctrl-C while it is working to pause: type a correction to steer it, or
press Enter to cancel the question.
"""

import argparse
import os
import sys

from labmate import agent, llm, profiles, report, ui
from labmate.session import HISTORY_MESSAGES, LOG_DIR, Session, clear_history, read_recent
from labmate.sources import openalex

_HELP = """Commands:
  /help      this list
  /clear     make the assistant forget the conversation so far
  /profile   show the model and tools in effect
  /spend     OpenAlex spend this session
  /quit      leave (Ctrl-D also works)
Press Ctrl-C while it is working to pause and steer it."""


def _parser():
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS so a flag given before the subcommand isn't overwritten by the
    # subcommand's own default.
    common.add_argument(
        "--profile", metavar="NAME", default=argparse.SUPPRESS,
        help="profile to use for this run (overrides LLM_PROFILE)",
    )
    common.add_argument(
        "-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
        help="hide progress lines",
    )
    parser = argparse.ArgumentParser(
        prog="labmate",
        parents=[common],
        description="A research lab mate: tests whether your idea already exists, finds "
        "related papers, and checks them for retractions. With no command, starts an "
        "interactive session.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    ask = sub.add_parser("ask", parents=[common], help="ask one question and print the answer")
    ask.add_argument("question", nargs="+", help="your question or research idea")
    sub.add_parser("profile", parents=[common], help="show the model and tools in effect")
    hist = sub.add_parser("history", parents=[common], help="show or clear the conversation history")
    hist.add_argument("-n", type=int, default=HISTORY_MESSAGES, metavar="N", help="messages to show")
    hist.add_argument("--clear", action="store_true", help="make the assistant forget the history")
    return parser


def _describe(profile):
    lines = [f"Model:   {llm.describe()}", f"Profile: {profiles.describe(profile)}", "Tools:"]
    for name, options in profile.tools.items():
        shown = ", ".join(f"{k}={v}" for k, v in options.items())
        lines.append(f"  {name}" + (f"  ({shown})" if shown else ""))
    return "\n".join(lines)


def _steer():
    """Ask the user how to redirect a paused question. A correction, or None to cancel."""
    ui.warn("\n[paused]")
    try:
        text = ui.ask("Type a correction to continue, or press Enter to cancel the question: ").strip()
    except (EOFError, KeyboardInterrupt):
        ui.warn("")
        return None
    return text or None


def _ask_prompt(question, choices):
    """Put the assistant's clarifying question to the user. Returns their answer
    (a number picks one of the choices; Enter means no preference), or None if
    input has ended. Ctrl-C is not caught here: it pauses the question."""
    ui.warn(f"\n[question] {question}")
    for number, choice in enumerate(choices, 1):
        ui.warn(f"  {number}) {choice}")
    hint = "a number or your own answer" if choices else "your answer"
    try:
        text = ui.ask(f"({hint}; Enter to let it decide) > ").strip()
    except EOFError:
        return None
    if not text:
        return "No preference: use your judgment and say what you assumed."
    if text.isdigit() and 1 <= int(text) <= len(choices):
        return choices[int(text) - 1]
    return text


def _history(args):
    if args.clear:
        clear_history()
        print("History cleared. The log file still has the full record.")
        return 0
    records = read_recent(os.path.join(LOG_DIR, "conversation.jsonl"), max(1, args.n))
    if not records:
        print("No history yet.")
        return 0
    for r in records:
        who = "you" if r["role"] == "user" else "labmate"
        print(f"[{r.get('ts', '?')}] {who}: {r['content']}\n")
    return 0


def _ask(args, profile):
    interactive = sys.stdin.isatty()  # someone can answer questions and steer
    session = Session(agent.system_prompt(profile, can_ask=interactive))
    try:
        text = agent.answer(
            session,
            " ".join(args.question),
            profile,
            steer=_steer if interactive else None,
            ask=_ask_prompt if interactive else None,
        )
    except Exception as exc:
        session.debug("error", message=str(exc))
        ui.warn(f"[error] {exc}")
        return 1
    if text is None:
        ui.warn("Stopped.")
        return 130
    ui.render(text + report.sources_table(text))
    return 0


def _slash(line, session, profile):
    """Run a /command. Returns "quit" to leave the session."""
    command = line.split()[0].lower()
    if command in ("/quit", "/exit"):
        return "quit"
    if command == "/help":
        print(_HELP)
    elif command == "/clear":
        session.clear()
        print("Forgotten. The log file still has the full record.")
    elif command == "/profile":
        print(_describe(profile))
    elif command == "/spend":
        print(f"OpenAlex spend this session: ${openalex.spent():.4f}")
    else:
        print(f"Unknown command {command}. Type /help.")
    return None


def _repl(profile):
    session = Session(agent.system_prompt(profile, can_ask=True))
    print(_describe(profile))
    if session.loaded:
        print(f"\nContinuing with the last {session.loaded} messages. (/clear to start fresh)")
    print("\nDescribe a research idea or ask about a topic. /help for commands, Ctrl-D to leave.\n")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n(Use /quit or Ctrl-D to leave.)")
            continue
        if not line:
            continue
        if line.lower() in ("quit", "exit") or line.startswith("/"):
            if line.startswith("/") and _slash(line, session, profile) != "quit":
                continue
            return 0
        try:
            text = agent.answer(session, line, profile, steer=_steer, ask=_ask_prompt)
        except Exception as exc:
            session.debug("error", message=str(exc))
            ui.warn(f"\n[error] {exc}\n")
            continue
        print()
        if text is None:
            print("(stopped)")
        else:
            ui.render(text + report.sources_table(text))
        print()
        ui.progress(f"(OpenAlex spend this session: ${openalex.spent():.4f})\n")


def main(argv=None):
    args = _parser().parse_args(argv)
    ui.set_quiet(getattr(args, "quiet", False))
    if args.command == "history":
        return _history(args)
    try:
        profile = agent.load_profile(getattr(args, "profile", None))
    except profiles.ProfileError as exc:
        ui.warn(f"[profile error] {exc}")
        return 1
    try:
        if args.command == "profile":
            print(_describe(profile))
            return 0
        if args.command == "ask":
            return _ask(args, profile)
        return _repl(profile)
    except KeyboardInterrupt:
        ui.warn("")
        return 130


if __name__ == "__main__":
    sys.exit(main())
