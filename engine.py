import pandas as pd
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import traceback
import uuid
import os

from groq_client import ask_groq
from utils import build_schema, extract_code

CHART_DIR = "/tmp/groq_charts"
os.makedirs(CHART_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are an expert Python data analyst.
A pandas DataFrame called `df` is already loaded in memory.
When the user asks a question:
1. Write ONLY a Python code block (no explanation outside it).
2. For charts: use matplotlib, save the figure to the variable `result_path`
   using plt.savefig(result_path, bbox_inches='tight') then plt.close().
3. For text/numeric answers: store the final answer in a variable called `answer`.
4. Do NOT call plt.show().
5. Do NOT import pandas — df is already available.
6. Use plotly only if the user explicitly asks for interactive charts.
7. Keep code concise and correct.
"""

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

    # Call Groq
    raw = ask_groq(messages, model=model)
    code = extract_code(raw)

    # Unique chart path per query
    chart_path = os.path.join(CHART_DIR, f"chart_{uuid.uuid4().hex[:8]}.png")

    # Sandbox: expose only what the code needs
    exec_env = {
        "df": df.copy(),
        "pd": pd,
        "plt": plt,
        "result_path": chart_path,
        "answer": None,
    }

    try:
        exec(compile(code, "<llm_code>", "exec"), exec_env)
    except Exception as e:
        # Retry once with the error fed back to the model
        fix_messages = messages + [
            {"role": "assistant", "content": f"```python\n{code}\n```"},
            {"role": "user",
             "content": f"That code raised an error:\n{e}\nPlease fix it."},
        ]
        raw2 = ask_groq(fix_messages, model=model)
        code = extract_code(raw2)
        try:
            exec(compile(code, "<llm_code_fixed>", "exec"), exec_env)
        except Exception as e2:
            return {
                "type": "error",
                "content": f"Error after retry:\n{e2}",
                "code": code,
                "raw": raw,
            }

    # Detect result type
    if os.path.exists(chart_path):
        return {"type": "chart", "content": chart_path, "code": code, "raw": raw}
    elif exec_env["answer"] is not None:
        return {"type": "text", "content": str(exec_env["answer"]), "code": code, "raw": raw}
    else:
        return {"type": "text", "content": "Done — no explicit answer returned.", "code": code, "raw": raw}