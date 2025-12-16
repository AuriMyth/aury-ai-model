from __future__ import annotations
from typing import AsyncIterator, TypeVar, Type
from pydantic import BaseModel, ConfigDict
from .types import Message, StreamEvent
from .providers.registry import make_adapter
from .providers.base import RequestFeatures
from .structured import StructuredView
from .context import aset_ctx, get_ctx
from .instrumentation import emit_start, emit_end, emit_event, set_usage, get_usage

T = TypeVar("T")

class ModelClient(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Connection-level config
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    headers: dict = {}
    transport: str | None = None  # e.g. "chat" | "responses" | provider-specific
    # Default generation parameters (can be overridden per-call)
    default_max_tokens: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_reasoning_effort: str | None = None  # "low" | "medium" | "high"

    def bind(self, **updates) -> "ModelClient":
        return self.model_copy(update=updates)

    def with_structured_output(self, schema: Type[T], steps: list | None=None) -> StructuredView[T]:
        return StructuredView(self, schema, steps)

    def with_retry(
        self,
        *,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 5.0,
        backoff_factor: float = 2.0,
        retry_on: tuple[type[BaseException], ...] | None = None,
        predicate: "callable | None" = None,
    ):
        """Return a tenacity-based retrying view of this client.

        Defaults to retrying on common transient model errors.
        """
        from .retry import RetryView, RetryConfig
        from .errors import (
            ModelTimeoutError, RateLimitError, ModelOverloadedError,
            TransportError, InvalidRequestError,
        )
        # If user didn't pass retry_on, use sensible defaults; ensure we DO NOT retry InvalidRequestError
        default_retry_on = (ModelTimeoutError, RateLimitError, ModelOverloadedError, TransportError)
        ro = retry_on or default_retry_on
        # If user accidentally included InvalidRequestError, filter it out
        ro = tuple(e for e in ro if e is not InvalidRequestError)
        cfg = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            retry_on=ro,
            predicate=predicate,
        )
        return RetryView(self, cfg)

    async def ainvoke(self, messages: list[Message], *, provider: str | None=None, model: str | None=None, headers: dict | None=None, **kw) -> Message:
        b = self._resolve_binding(provider, model, headers)
        metrics = emit_start(b["provider"], b["model"])
        async with aset_ctx(provider=b["provider"], model=b["model"], extra_headers=b["headers"]):
            adapter = make_adapter(**b)
            req = self._features_from(messages, **kw)
            try:
                # 避免上一次请求的 usage “串台”
                set_usage(None)
                m = await adapter.ainvoke(messages, req, **kw)
                u = get_usage()
                emit_end(metrics, usage=(u.model_dump() if u else None))
                return m
            except Exception as e:
                emit_end(metrics, error=e)
                raise

    async def astream(self, messages: list[Message], *, provider: str | None=None, model: str | None=None, headers: dict | None=None, yield_usage_event: bool=False, **kw) -> AsyncIterator[StreamEvent]:
        b = self._resolve_binding(provider, model, headers)
        metrics = emit_start(b["provider"], b["model"])
        async with aset_ctx(provider=b["provider"], model=b["model"], extra_headers=b["headers"]):
            adapter = make_adapter(**b)
            req = self._features_from(messages, **kw)
            usage: dict | None = None
            try:
                # 避免上一次请求的 usage “串台”
                set_usage(None)
                async for ev in adapter.astream(messages, req, **kw):
                    if ev.type == "usage" and ev.usage:
                        usage = dict(
                            input_tokens=ev.usage.input_tokens,
                            output_tokens=ev.usage.output_tokens,
                            reasoning_tokens=ev.usage.reasoning_tokens,
                            total_tokens=ev.usage.total_tokens,
                        )
                        set_usage(ev.usage)
                        if yield_usage_event:
                            emit_event(ev.type, ev.model_dump())
                            yield ev
                        continue
                    emit_event(ev.type, ev.model_dump())
                    yield ev
                # 某些 provider 不会显式产出 usage 事件，但会 set_usage()
                if usage is None:
                    u = get_usage()
                    if u is not None:
                        usage = dict(
                            input_tokens=u.input_tokens,
                            output_tokens=u.output_tokens,
                            reasoning_tokens=u.reasoning_tokens,
                            total_tokens=u.total_tokens,
                        )
                emit_end(metrics, usage=usage)
            except Exception as e:
                emit_end(metrics, error=e)
                raise

    def _resolve_binding(self, provider: str | None, model: str | None, headers_override: dict | None=None) -> dict:
        p = provider or self.provider
        m = model or self.model
        if not p or not m:
            raise ValueError("provider/model must be set via client.bind(...) or call override.")
        merged_headers = {**self.headers, **get_ctx().extra_headers}
        if headers_override:
            merged_headers.update(headers_override)
        return {
            "provider": p,
            "model": m,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "headers": merged_headers,
            "transport": self.transport,
        }

    def _features_from(self, messages: list[Message], **kw) -> RequestFeatures:
        from .types import Image
        has_images = any(isinstance(p, Image) for m in messages for p in m.parts)
        # Merge defaults with call-time overrides (call-time takes precedence)
        return RequestFeatures(
            # Feature flags
            needs_tools=bool(kw.get("tools")),
            needs_structured=bool(kw.get("text_format") or kw.get("response_format") or kw.get("schema")),
            return_thinking=bool(kw.get("return_thinking")),
            has_images=has_images,
            # Common generation parameters (call-time > bind-time defaults)
            max_tokens=kw.get("max_tokens") if kw.get("max_tokens") is not None else self.default_max_tokens,
            max_completion_tokens=kw.get("max_completion_tokens"),
            temperature=kw.get("temperature") if kw.get("temperature") is not None else self.default_temperature,
            top_p=kw.get("top_p") if kw.get("top_p") is not None else self.default_top_p,
            stop=kw.get("stop"),
            seed=kw.get("seed"),
            reasoning_effort=kw.get("reasoning_effort") or self.default_reasoning_effort,
            # Structured output
            response_format=kw.get("response_format"),
            text_format=kw.get("text_format"),
            # Provider-specific
            previous_response_id=kw.get("previous_response_id"),
            caching=kw.get("caching"),
            extra_body=kw.get("extra_body"),
        )

    def last_usage(self):
        from .instrumentation import get_usage
        return get_usage()
