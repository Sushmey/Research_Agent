"""suggest_queries: turn an idea into several keyword queries."""

from research import query_planner

SCHEMA = {
    "type": "function",
    "function": {
        "name": "suggest_queries",
        "description": (
            "Turn a research idea or topic into 4-6 short keyword queries from "
            "different angles (the whole idea, the method alone, the problem alone, "
            "synonyms, an adjacent field). Call this FIRST for any new idea, before "
            "search_papers, then run the queries it returns. Call it again when you "
            "learn new terminology from a paper and want queries for it. Search "
            "is keyword-based, so one phrasing alone misses papers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "The idea or topic in a sentence or two, self-contained "
                        "(include context from earlier in the conversation)."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
}


def prompt_hint(options):
    return (
        "For a new idea, call suggest_queries first, then run the queries it returns "
        "as separate search_papers calls in one turn. Prefer those queries over writing "
        "your own, and call suggest_queries again when a paper introduces terminology "
        "you hadn't thought of."
    )


def run(topic, options=None):
    planned = query_planner.plan(topic)
    if not planned:
        return {"queries": [], "note": "No queries could be generated; write your own keyword queries."}
    return planned
