# utils/azure_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")

def is_azure_configured():
    return bool(AZURE_ENDPOINT and AZURE_KEY and AZURE_DEPLOYMENT)

def synthesize_with_azure(system_prompt: str, user_prompt: str, context_docs: list, max_tokens: int = 300):
    """
    Calls Azure OpenAI chat completions (chat-style) with given prompts and retrieved context.
    Returns text or raises on HTTP error.
    """
    if not is_azure_configured():
        return None

    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {
        "api-key": AZURE_KEY,
        "Content-Type": "application/json"
    }

    # Build messages; include context documents in user message
    context_text = "\n\n".join(context_docs) if context_docs else ""
    user_content = user_prompt + "\n\nContext:\n" + context_text

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Azure returns choices[0].message.content
    return data["choices"][0]["message"]["content"]
