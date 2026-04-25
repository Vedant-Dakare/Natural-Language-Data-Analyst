import re
import pandas as pd

def build_schema(df: pd.DataFrame) -> str:
    lines = [f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns", "Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(4).tolist()
        lines.append(f"  - {col} ({dtype}): e.g. {sample}")
    return "\n".join(lines)


def is_valid_python(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except Exception:
        return False


def extract_code(text: str) -> str | None:
    """Extract ONLY valid Python code block from LLM response."""

    # 1. ```python block
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. generic ``` block
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    
    return None