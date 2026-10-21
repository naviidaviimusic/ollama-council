"""Adaptive multi-model council for local Ollama models."""

from .council import DEFAULT_MEMBERS, Council, Result, ask

__all__ = ["Council", "Result", "ask", "DEFAULT_MEMBERS"]
__version__ = "0.1.0"
