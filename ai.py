import os
from .knowledge import SYSTEM, offline

def _get(secrets, name, default=""):
    try:
        v = secrets.get(name, "")
    except Exception:
        v = ""
    return v or os.getenv(name, default)

def ask(mode, history, secrets):
    """history: [{'role': 'user'|'assistant', 'content': str}]"""
    key = _get(secrets, "GEMINI_API_KEY")
    if not key:
        return offline(mode, history[-1]["content"])
    try:
        from google import genai
        from google.genai import types
        model = _get(secrets, "GEMINI_MODEL", "gemini-2.5-flash")
        contents = [types.Content(role="user" if m["role"] == "user" else "model",
                                  parts=[types.Part(text=m["content"])]) for m in history[-12:]]
        r = genai.Client(api_key=key).models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM[mode], max_output_tokens=900))
        return r.text or offline(mode, history[-1]["content"])
    except Exception as e:
        return f"Gemini unavailable ({type(e).__name__}). Fallback:\n\n" + offline(mode, history[-1]["content"])
