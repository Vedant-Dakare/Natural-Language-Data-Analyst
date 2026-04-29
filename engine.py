import json
import re
import pandas as pd

from groq_client import ask_groq
from utils import build_schema

SYSTEM_PROMPT = """You are an advanced AI Data Analyst and Visualization Designer.

Your goal is NOT just to answer queries, but to present results in a visually rich, insightful, and professional way that is easy to understand and impressive to users.

You are given:
1) A pandas DataFrame schema
2) A user query

You must return ONLY valid JSON (no explanation, no markdown).

OUTPUT FORMAT:
{
  "type": "chart" | "table" | "graph",
  "title": "Clear and meaningful title",
  "description": "Short 1-2 line insight explaining the result",
  "data": {...},
  "graph": {
    "nodes": [],
    "edges": []
  },
  "style": {
    "chart_type": "bar" | "line" | "histogram",
    "color_theme": "modern",
    "highlight": "important trend or value",
    "x_label": "",
    "y_label": ""
  }
}

RULES:
1. Do NOT return plain/raw data only. Always include title, description, and structured format.
2. Relationships/hierarchy => type="graph".
3. Comparisons/trends => type="chart".
4. Raw data => type="table" only when necessary.
5. Graph nodes must be meaningful/readable with descriptive labels.
6. Graph edges must describe relationships like belongs_to / has_sales / related_to.
7. For graph output, represent the full dataset structure with a root dataset node, one node per column, and summary/value nodes where appropriate. Never return an empty graph.
8. Chart selection:
   - time => line chart
   - category comparison => bar chart
   - distribution => histogram
9. Always include axis labels for charts.
10. Description must explain a key finding and never be generic.
11. Never return text outside JSON.
"""


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract first JSON object from noisy responses
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _normalize_result(parsed: dict | None, raw: str) -> dict:
    if not parsed:
        return {
            "type": "error",
            "content": "Model returned invalid JSON output.",
            "raw": raw,
            "json": None,
        }

    result_type = parsed.get("type", "table")
    if result_type not in {"chart", "table", "graph"}:
        result_type = "table"

    parsed.setdefault("title", "Data Analysis Result")
    parsed.setdefault("description", "Generated insight based on your query.")
    parsed.setdefault("data", {})
    parsed.setdefault("graph", {"nodes": [], "edges": []})
    parsed.setdefault(
        "style",
        {
            "chart_type": "bar",
            "color_theme": "modern",
            "highlight": "",
            "x_label": "",
            "y_label": "",
        },
    )

    return {
        "type": result_type,
        "content": parsed,
        "raw": raw,
        "json": parsed,
    }

def build_messages(
    schema: str,
    question: str,
    history: list[dict],  
) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Here is the DataFrame schema:\n{schema}\n\nRemember this for all questions."
        },
        {
            "role": "assistant",
            "content": "Understood. I have the schema. Ask your question."
        },
    ]
    # Add previous turns for multi-turn memory (last 6 exchanges)
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": question})
    return messages


def run_query(
    df: pd.DataFrame,
    question: str,
    history: list[dict],
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    schema = build_schema(df)
    messages = build_messages(schema, question, history)

    try:
        raw = ask_groq(messages, model=model)
    except Exception as exc:
        return {
            "type": "error",
            "content": f"The model is currently busy (rate limited). Please try again in a few seconds. Details: {exc}",
            "raw": str(exc),
            "json": None,
        }

    parsed = _extract_json(raw)

    if parsed is None:
        fix_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": "The output was not valid JSON. Return ONLY valid JSON that matches the required schema.",
            },
        ]
        try:
            raw_retry = ask_groq(fix_messages, model=model)
        except Exception as exc:
            return {
                "type": "error",
                "content": f"The model is currently busy (rate limited). Please try again in a few seconds. Details: {exc}",
                "raw": str(exc),
                "json": None,
            }
        parsed = _extract_json(raw_retry)
        return _normalize_result(parsed, raw_retry)

    return _normalize_result(parsed, raw)