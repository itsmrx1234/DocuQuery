"""Small HTTP client for a locally running Ollama server."""
import requests


class OllamaError(RuntimeError):
    pass


def generate(base_url: str, model: str, prompt: str) -> str:
    try:
        response = requests.post(f"{base_url.rstrip('/')}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except requests.RequestException as exc:
        raise OllamaError("Ollama is unavailable. Start Ollama and pull the configured model.") from exc
    if not answer:
        raise OllamaError("Ollama returned an empty answer. Confirm the configured model is installed.")
    return answer
