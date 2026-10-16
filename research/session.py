"""The conversation history the model sees, and the logs on disk.

Two files in logs/ at the repo root, both append-only and kept across runs:

    conversation.jsonl   every question and every final answer, from every
                         session. It is the full record. Only the most recent
                         HISTORY_MESSAGES of it are sent to the model (a
                         sliding window), so a restart picks up where the last
                         run left off without growing the prompt.
    tools.jsonl          every model reply, tool call, tool result, refusal and
                         error, for debugging. It is written but never read
                         back, and the model never sees it.

While a question is being answered, the model also sees that turn's own tool
calls and results (it needs them to write the answer), held in memory. They
are not part of the history: once the answer is written, only the question and
the answer are.

The logs contain whatever was typed and every tool result, so keep logs/ out of
version control.
"""

import datetime
import json
import os
import threading
from collections import deque

HISTORY_MESSAGES = 10  # most recent messages (questions and answers) the model sees
MAX_PAST_CHARS = 2000  # one old message is clipped to this, so a long answer can't fill the context

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class _JsonlFile:
    """Append-only JSON-lines file. A write failure warns once and then the
    session carries on without that file."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()  # tool calls run in threads
        self._failed = False

    def write(self, record):
        if self._failed:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            self._failed = True
            print(f"[log] can't write {self.path}: {exc}. Continuing without it.")


def _read_recent(path, count):
    """The last `count` user/assistant messages in a conversation file. Reads
    the whole file but keeps only the tail; unreadable lines are skipped."""
    recent = deque(maxlen=count)
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("role") in ("user", "assistant") and isinstance(r.get("content"), str):
                    recent.append({"role": r["role"], "content": r["content"]})
    except OSError:
        pass
    return list(recent)


class Session:
    def __init__(self, system, log_dir=LOG_DIR, window=HISTORY_MESSAGES):
        self.system = system
        self.window = window
        self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._conversation = _JsonlFile(os.path.join(log_dir, "conversation.jsonl"))
        self._tools = _JsonlFile(os.path.join(log_dir, "tools.jsonl"))
        # Only the tail is needed in memory; the file keeps everything.
        self._recent = _read_recent(self._conversation.path, window)
        self.loaded = len(self._recent)  # messages carried over from earlier runs
        self._turn = []  # this question's messages, including its tool calls
        self._turn_number = 0

    # --- what the model sees -------------------------------------------------

    def _past(self):
        past = self._recent[-self.window :]
        while past and past[0]["role"] != "user":  # the model expects to start on a user message
            past = past[1:]
        return [
            {
                "role": m["role"],
                "content": m["content"]
                if len(m["content"]) <= MAX_PAST_CHARS
                else m["content"][:MAX_PAST_CHARS].rstrip() + " ...",
            }
            for m in past
        ]

    def messages(self, extra=()):
        """System prompt, the last `window` messages of past conversation, then this turn."""
        return [{"role": "system", "content": self.system}] + self._past() + self._turn + list(extra)

    # --- one question --------------------------------------------------------

    def begin_turn(self, question):
        """Start a question. The window shown to the model is fixed here, so it
        holds only earlier conversation, not this question."""
        if self._turn:  # the previous question failed before it was answered; keep it in the history
            self._recent = (self._recent + [self._turn[0]])[-self.window :]
        self._turn_number += 1
        self._turn = [{"role": "user", "content": question}]
        self._conversation.write({"ts": _now(), "session": self.id, "role": "user", "content": question})
        self.debug("question", content=question)

    def add(self, *messages):
        """Add the model's replies and tool results for this turn (memory only)."""
        self._turn.extend(messages)

    def finish_turn(self, answer):
        """Record the answer. The question and answer join the history the
        next turn's window is taken from."""
        self._conversation.write({"ts": _now(), "session": self.id, "role": "assistant", "content": answer})
        self._recent.append(self._turn[0])
        self._recent.append({"role": "assistant", "content": answer})
        self._recent = self._recent[-self.window :]
        self._turn = []

    # --- debug log -----------------------------------------------------------

    def debug(self, event, **data):
        self._tools.write({"ts": _now(), "session": self.id, "turn": self._turn_number, "event": event, **data})
