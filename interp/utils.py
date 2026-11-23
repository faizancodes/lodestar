"""Shared utilities for Lodestar's Python backend: logging, filesystem safety, model helpers, and SSE events."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring a default stderr handler once if none is attached.

    Args:
        name: Module logger name (typically ``__name__``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    global _configured
    root = logging.getLogger("interp")
    if not _configured and not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)
        root.setLevel(os.environ.get("LODESTAR_LOG_LEVEL", "INFO").upper())
        _configured = True
    return logging.getLogger(name)


def sse_event(event_type: str, **payload: Any) -> dict[str, Any]:
    """Construct a server-sent event payload dictionary.

    Args:
        event_type: Discriminator stored under the ``"type"`` key.
        **payload: Additional JSON-serializable event fields.

    Returns:
        Dictionary suitable for SSE serialization.
    """
    return {"type": event_type, **payload}


def model_short_name(model_id: str) -> str:
    """Return the last path segment of a Hugging Face model ID or local model directory."""
    return str(model_id).rstrip("/").split("/")[-1]


def slug(model_id: str) -> str:
    """Convert a model ID or local path into a filesystem-safe directory or file stem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_short_name(model_id)) or "model"


def text_config_of(config: Any) -> Any:
    """Resolve the text sub-config for multimodal models, falling back to ``config`` itself."""
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", None) or config


def unwrap_tensor(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Extract the primary tensor from a PyTorch module output (which may be a tuple)."""
    return output[0] if isinstance(output, tuple) else output


def empty_device_cache() -> None:
    """Release cached allocator memory on MPS or CUDA when available."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text`` via a same-directory temporary file.

    A crash mid-write leaves the previous file intact instead of a truncated half-file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any, *, indent: int | None = 1, ensure_ascii: bool = False) -> None:
    """Atomically serialize ``data`` as JSON to ``path``."""
    atomic_write_text(path, json.dumps(data, indent=indent, ensure_ascii=ensure_ascii))
