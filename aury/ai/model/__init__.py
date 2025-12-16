"""auri-ai-core model 模块

统一、高可靠的 LLM 调用层。
"""

__version__ = "0.1.0"

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
