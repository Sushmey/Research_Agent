# labmate

A command-line research lab mate. Describe a research idea and it checks whether the idea
already exists in the literature, finds related papers, reads their conclusions and
future-work sections, and checks every paper for retractions before suggesting it.

## Setup

Needs Python 3.11+.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                  # add ".[anthropic]" to use Claude
cp .env.example .env              # then edit it
```

In `.env`:

- **Model.** Defaults to a local Ollama (`gpt-oss:20b`). To use a hosted model, set `LLM_PROVIDER`
  (`openai_compatible` or `anthropic`), `LLM_MODEL` and `LLM_API_KEY`.
- **`OPENALEX_API_KEY`.** A free key from <https://openalex.org/settings/api>. Click the activation
  link in the email, or requests fail with `key_not_activated`.

## Run

```
labmate                              # interactive session
labmate ask "has anyone tried X?"    # one question, answer on stdout
labmate profile                      # model, profile and tools in effect
labmate history                      # recent conversation (--clear to forget it)
```

Options: `--profile NAME` (use a profile for this run), `-q` (hide progress lines).
Without installing: `python3 main.py`.

In a session: `/help`, `/clear`, `/profile`, `/spend`, `/quit`. Press **Ctrl-C** while it is working
to pause, then type a correction to steer it or press Enter to cancel.

## Profiles

`small` (the default for Ollama) gives a local model more scaffolding and tighter limits;
`large` suits hosted models. Set `LLM_PROFILE=small|large`, or define your own in
`profiles.toml` (see `profiles.example.toml`).

## Notes

- Answers end with a "Papers mentioned" table: publication date, age and retraction status,
  built from what the tools returned.
- Full text is read only for papers with an arXiv HTML version.
- Conversations and tool activity are logged to `logs/` (gitignored).
