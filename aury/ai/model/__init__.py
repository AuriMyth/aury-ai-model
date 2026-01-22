"""auri-ai-core model 模块

统一、高可靠的 LLM 调用层。
"""
import logging
from typing import Any

__version__ = "0.1.0"

# =============================================================================
# TRACE Level 支持
# =============================================================================
# TRACE (5) < DEBUG (10)，用于超细粒度调试（如每个 streaming chunk）
# 如果应用端使用 aury-boot，TRACE 已经注册；否则这里注册
TRACE = 5
if logging.getLevelName(TRACE) == "Level 5":
    logging.addLevelName(TRACE, "TRACE")


def _ensure_trace_method() -> None:
    """确保 logging.Logger 有 trace() 方法。"""
    if hasattr(logging.Logger, "trace"):
        return
    
    def trace(self: logging.Logger, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)
    
    logging.Logger.trace = trace  # type: ignore[attr-defined]

_ensure_trace_method()

# 统一 logger
logger = logging.getLogger("aury.ai.model")

from .client import ModelClient
from .types import (
    Message, StreamEvent, Usage, ToolCall,
    Text, Image, Thinking, FileRef, Part, Evt, Role, msg, StreamCollector,
)
from .tools import ToolSpec, ToolKind, FunctionToolSpec, MCPToolSpec, BuiltinToolSpec
from .context import RequestContext, get_ctx, set_ctx, aset_ctx, push_ctx, pop_ctx, model_ctx
from .errors import (
    ModelError, ModelTimeoutError, RateLimitError, ModelOverloadedError,
    InvalidRequestError, TransportError, SchemaMismatchError, StreamBrokenError,
    ProviderNotInstalledError,
)
from .instrumentation import (
    InstrumentSink, RequestMetrics, register_sink, clear_sinks,
)

# re-export retry view type for typing (optional)
try:
    from .retry import RetryView  # noqa: F401
except Exception:
    pass

__all__ = [
    # version
    "__version__",
    # client
    "ModelClient",
    # types
    "Message", "StreamEvent", "Usage", "ToolCall",
    "Text", "Image", "Thinking", "FileRef", "Part", "Evt", "Role", "msg", "StreamCollector",
    # tools
    "ToolSpec", "ToolKind", "FunctionToolSpec", "MCPToolSpec", "BuiltinToolSpec",
    # context
    "RequestContext", "get_ctx", "set_ctx", "aset_ctx", "push_ctx", "pop_ctx", "model_ctx",
    # errors
    "ModelError", "ModelTimeoutError", "RateLimitError", "ModelOverloadedError",
    "InvalidRequestError", "TransportError", "SchemaMismatchError", "StreamBrokenError",
    "ProviderNotInstalledError",
    # instrumentation
    "InstrumentSink", "RequestMetrics", "register_sink", "clear_sinks",
    # retry
    "RetryView",
]
