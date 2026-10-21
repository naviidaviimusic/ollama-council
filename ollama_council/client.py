"""Minimal Ollama HTTP client (stdlib only).

Generation is always streamed: when the deadline passes we close the socket and
Ollama aborts the request. With stream=False the server keeps generating for a
client that already gave up, and abandoned requests pile up into a backlog.
"""

import json
import os
import socket
import time
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL


class OllamaError(RuntimeError):
    pass


class OllamaTimeout(OllamaError):
    pass


def _post(path: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def is_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def installed() -> list:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []


def generate(prompt: str, model: str, system: str = None, max_tokens: int = 800,
             temperature: float = 0.7, fmt=None, timeout: float = 120,
             keep_alive: str = "2m") -> str:
    """Return the model's full answer. Raises OllamaTimeout / OllamaError."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    if system:
        payload["system"] = system
    if fmt:
        payload["format"] = fmt
    deadline = time.monotonic() + timeout
    chunks = []
    try:
        with _post("/api/generate", payload, timeout) as r:
            for line in r:
                if not line.strip():
                    continue
                part = json.loads(line)
                if part.get("error"):
                    raise OllamaError(part["error"])
                chunks.append(part.get("response", ""))
                if part.get("done"):
                    break
                if time.monotonic() > deadline:
                    # leaving the with-block closes the stream → generation is cancelled
                    raise OllamaTimeout(f"{model} exceeded {timeout:.0f}s")
    except (TimeoutError, socket.timeout) as e:
        raise OllamaTimeout(f"{model} exceeded {timeout:.0f}s") from e
    except urllib.error.HTTPError as e:
        raise OllamaError(f"{model}: HTTP {e.code} {e.read()[:200]!r}") from e
    except urllib.error.URLError as e:
        if "timed out" in str(e).lower():
            raise OllamaTimeout(f"{model} exceeded {timeout:.0f}s") from e
        raise OllamaError(f"Ollama unreachable at {OLLAMA_URL}: {e.reason}") from e
    return "".join(chunks).strip()


def embed(text: str, model: str = "nomic-embed-text", timeout: float = 30):
    """Embedding vector, or None if the embed model isn't available."""
    try:
        with _post("/api/embeddings", {"model": model, "prompt": text[:4000],
                                       "keep_alive": "5m"}, timeout) as r:
            return json.loads(r.read()).get("embedding") or None
    except Exception:
        return None
