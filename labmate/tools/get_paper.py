"""get_paper: fetch one paper by OpenAlex ID or DOI."""

from labmate.sources import openalex

SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_paper",
        "description": (
            "Fetch one paper by OpenAlex ID (like 'W2741809807') or DOI, with "
            "its full abstract and retraction status. Free. Use it to read a "
            "paper more closely or to compare two papers you have seen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "OpenAlex ID or DOI."},
            },
            "required": ["id"],
        },
    },
}


def prompt_hint(options):
    return "Use get_paper to read a paper more closely or to compare two papers."


def run(id, options=None):
    return openalex.get_work(id)
