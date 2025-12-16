from __future__ import annotations
from typing import Protocol
from pydantic import BaseModel, Field, ConfigDict
from time import monotonic
from contextvars import ContextVar
from .context import get_ctx
from .types import Usage

class RequestMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: str
    model: str
    start_ms: float
    end_ms: float | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None

class InstrumentSink(Protocol):
    def on_request_start(self, metrics: RequestMetrics): ...
    def on_stream_event(self, event_type: str, payload: dict): ...
    def on_request_end(self, metrics: RequestMetrics): ...
    def on_error(self, metrics: RequestMetrics): ...

class _NoopSink:
    def on_request_start(self, metrics: RequestMetrics): ...
    def on_stream_event(self, event_type: str, payload: dict): ...
    def on_request_end(self, metrics: RequestMetrics): ...
    def on_error(self, metrics: RequestMetrics): ...

_sinks: list[InstrumentSink] = [_NoopSink()]
# Per-request latest usage capture (accessible to callers)
_usage_var: ContextVar[Usage | None] = ContextVar("usage_var", default=None)

def register_sink(sink: InstrumentSink):
    _sinks.append(sink)

def clear_sinks():
    global _sinks
    _sinks = [_NoopSink()]

def set_usage(u: Usage | None):
    _usage_var.set(u)

def get_usage() -> Usage | None:
    return _usage_var.get()

class Timer:
    __slots__ = ("_t0",)
    def __init__(self):
        self._t0 = monotonic()
    def elapsed_ms(self) -> float:
        return (monotonic() - self._t0) * 1000.0

def emit_start(provider: str, model: str) -> RequestMetrics:
    m = RequestMetrics(provider=provider, model=model, start_ms=monotonic()*1000.0)
    for s in _sinks:
        s.on_request_start(m)
    return m

def emit_event(event_type: str, payload: dict):
    for s in _sinks:
        s.on_stream_event(event_type, payload)

def emit_end(metrics: RequestMetrics, *, usage: dict | None=None, error: Exception | None=None):
    end_ms = monotonic()*1000.0
    latency = end_ms - metrics.start_ms
    m = metrics.model_copy(update={"end_ms": end_ms, "latency_ms": latency})
    if usage:
        m = m.model_copy(update={
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })
    if error:
        m = m.model_copy(update={"error": str(error)})
        for s in _sinks:
            s.on_error(m)
    else:
        for s in _sinks:
            s.on_request_end(m)
