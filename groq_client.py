import os
from groq import Groq
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
) -> str:
    """
    messages: list of {"role": "system"/"user"/"assistant", "content": "..."}
    Returns the assistant's reply as a plain string.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()