import os
import re
import time
from groq import Groq
from groq import RateLimitError
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELS = {
    "LLaMA 3.3 70B (best)":   "llama-3.3-70b-versatile",
    "LLaMA 3.1 8B (fastest)":  "llama-3.1-8b-instant",
    "Mixtral 8x7B":             "mixtral-8x7b-32768",
    "Gemma 2 9B":               "gemma2-9b-it",
}

def ask_groq(
    messages: list[dict],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    retries: int = 4,
) -> str:
    """
    messages: list of {"role": "system"/"user"/"assistant", "content": "..."}
    Returns the assistant's reply as a plain string.
    """
    max_tokens_local = max_tokens
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens_local,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as exc:
            if attempt >= retries:
                raise

            # Parse provider hint like "Please try again in 620ms" when present.
            message = str(exc)
            retry_ms_match = re.search(r"try again in\s+(\d+)ms", message, flags=re.IGNORECASE)
            wait_seconds = 0.9
            if retry_ms_match:
                wait_seconds = max(float(retry_ms_match.group(1)) / 1000.0, 0.25)

            wait_seconds += 0.35 * attempt
            max_tokens_local = max(256, int(max_tokens_local * 0.85))
            time.sleep(wait_seconds)

    raise RuntimeError("Groq request failed after retries.")