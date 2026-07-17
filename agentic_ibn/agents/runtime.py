from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def build_groq_agent(*, model_id: str, **agent_kwargs: Any) -> Any:
    """Create an Agno Agent lazily so deterministic tests do not require Agno."""
    try:
        from agno.agent import Agent
        from agno.models.groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "Agno/Groq dependencies are not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return Agent(model=Groq(id=model_id), **agent_kwargs)


def coerce_response_model(response: Any, schema: type[T]) -> T:
    """Convert Agno RunOutput.content into the requested Pydantic schema."""
    content = getattr(response, "content", response)
    if isinstance(content, schema):
        return content
    if isinstance(content, BaseModel):
        return schema.model_validate(content.model_dump())
    if isinstance(content, dict):
        return schema.model_validate(content)
    if isinstance(content, str):
        cleaned = content.replace("```json", "").replace("```", "").strip()
        return schema.model_validate(json.loads(cleaned))
    raise TypeError(f"Unsupported Agno response content: {type(content)!r}")
