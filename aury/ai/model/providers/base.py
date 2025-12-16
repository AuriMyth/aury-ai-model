from __future__ import annotations
from typing import Protocol, AsyncIterator
from pydantic import BaseModel
from ..types import Message, StreamEvent

class RequestFeatures(BaseModel):
    """Request features and common parameters.
    
    Common parameters (cross-provider):
    - max_tokens: Maximum tokens to generate
    - temperature: Sampling temperature (0-2)
    - top_p: Nucleus sampling probability
    - stop: Stop sequences
    - reasoning_effort: Thinking effort level ("low"/"medium"/"high")
    
    Provider-specific parameters:
    - extra_body: Dict passed directly to provider API (for provider-specific options)
    """
    # Feature flags (internal)
    needs_tools: bool = False
    needs_structured: bool = False
    return_thinking: bool = False
    has_images: bool = False
    
    # Common generation parameters
    max_tokens: int | None = None
    max_completion_tokens: int | None = None  # OpenAI o-series / newer models
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    reasoning_effort: str | None = None  # "low" | "medium" | "high"
    
    # Structured output
    response_format: dict | None = None  # Chat API
    text_format: dict | None = None  # Responses API / Doubao
    
    # Provider-specific
    previous_response_id: str | None = None  # Doubao/OpenAI Responses
    caching: dict | None = None  # Doubao caching config
    extra_body: dict | None = None  # Pass-through to provider API

class ProviderRoute(BaseModel):
    channel: str
    reason: str

class ProviderAdapter(Protocol):
    name: str
    model: str
    def route(self, req: RequestFeatures) -> ProviderRoute: ...
    async def ainvoke(self, messages: list[Message], req: RequestFeatures, **kw) -> Message: ...
    async def astream(self, messages: list[Message], req: RequestFeatures, **kw) -> AsyncIterator[StreamEvent]: ...
