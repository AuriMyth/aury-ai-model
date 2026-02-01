from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, TypeAlias
import json
import os
import base64
import mimetypes

class Evt(StrEnum):
    content = "content"
    thinking = "thinking"
    thinking_completed = "thinking_completed"  # 思考完成
    tool_call_start = "tool_call_start"      # 工具调用开始（首次通知）
    tool_call_delta = "tool_call_delta"      # 工具参数增量
    tool_call_progress = "tool_call_progress"  # 工具参数接收进度
    tool_call = "tool_call"                  # 工具调用完整（参数完整）
    image = "image"                          # 图片输出（如 Gemini 图片生成）
    usage = "usage"
    completed = "completed"
    error = "error"

class Role(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"

class Text(BaseModel):
    type: Literal["text"] = "text"
    text: str

class Image(BaseModel):
    type: Literal["image_url"] = "image_url"
    url: str

    @property
    def data(self) -> bytes | None:
        """Extract raw image bytes from base64 data URL.
        Returns None if URL is not a data URL.
        """
        if self.url and self.url.startswith("data:") and ";base64," in self.url:
            base64_part = self.url.split(";base64,", 1)[1]
            return base64.b64decode(base64_part)
        return None

    @property
    def mime_type(self) -> str | None:
        """Extract MIME type from data URL (e.g., 'image/png').
        Returns None if URL is not a data URL.
        """
        if self.url and self.url.startswith("data:") and ";" in self.url:
            # data:image/png;base64,... -> image/png
            return self.url.split(":", 1)[1].split(";", 1)[0]
        return None

class Thinking(BaseModel):
    type: Literal["thinking"] = "thinking"
    text: str

class FileRef(BaseModel):
    type: Literal["file_ref"] = "file_ref"
    id: str

Part: TypeAlias = Text | Image | Thinking | FileRef

class ToolCall(BaseModel):
    id: str
    name: str
    arguments_json: str
    mcp_server_id: str | None = None

    @property
    def arguments(self) -> dict:
        """Parsed arguments as dict (convenience).
        Safe fallback to empty dict if JSON is invalid.
        """
        try:
            return json.loads(self.arguments_json or "{}")
        except Exception:
            return {}

    def with_arguments(self, args: dict) -> "ToolCall":
        """Return a copy with arguments_json set from dict.
        This does not mutate the current instance.
        """
        try:
            args_json = json.dumps(args)
        except Exception:
            args_json = "{}"
        return self.model_copy(update={"arguments_json": args_json})

class Message(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Role
    parts: list[Part] = Field(default_factory=list)
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict = Field(default_factory=dict)
    # OpenRouter: reasoning_details must be preserved and passed back for tool calling
    reasoning_details: list[dict] | None = None

    @property
    def text(self) -> str:
        """Extract all text content from parts (Text and Thinking)."""
        return "".join(p.text for p in self.parts if hasattr(p, "text"))

    @property
    def content(self) -> str:
        """Alias for text. Extract all text content from parts."""
        return self.text

    @property
    def thinking(self) -> str:
        """Extract thinking content only."""
        return "".join(p.text for p in self.parts if isinstance(p, Thinking))

    @property
    def has_tool_calls(self) -> bool:
        """Check if message has tool calls."""
        return bool(self.tool_calls)

    @property
    def first_tool_call(self) -> ToolCall | None:
        """Get first tool call if any."""
        return self.tool_calls[0] if self.tool_calls else None

    def __str__(self) -> str:
        """String representation for debugging."""
        return self.text or f"[{self.role}]"

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False

class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Evt
    delta: str | None = None
    tool_call: ToolCall | None = None
    tool_call_delta: dict | None = None      # 工具参数增量: {"call_id": str, "arguments_delta": str}
    tool_call_progress: dict | None = None   # 工具参数进度: {"call_id": str, "bytes_received": int, "last_delta_size": int}
    image: Image | None = None               # 图片输出
    usage: Usage | None = None
    error: str | None = None
    # OpenRouter: reasoning_details for Gemini 3 tool calling (emitted with completed event)
    reasoning_details: list[dict] | None = None

class msg:  # convenience constructors
    @staticmethod
    def system(text: str) -> Message:
        return Message(role=Role.system, parts=[Text(text=text)])

    @staticmethod
    def _process_image_path(path: str) -> str:
        """
        处理图片路径：
        - 如果是 HTTP/HTTPS URL，直接返回
        - 如果是本地文件路径，读取并转换为 base64 data URL
        """
        # 判断是否是 HTTP/HTTPS URL
        if path.startswith(('http://', 'https://')):
            return path
        
        # 判断是否是本地文件路径
        if os.path.isfile(path):
            # 读取文件
            with open(path, 'rb') as f:
                image_data = f.read()
            
            # 获取 MIME 类型
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type or not mime_type.startswith('image/'):
                # 默认使用 image/jpeg
                mime_type = 'image/jpeg'
            
            # 转换为 base64
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            # 返回 data URL
            return f"data:{mime_type};base64,{base64_data}"
        
        # 如果既不是 URL 也不是有效的本地文件，直接返回原始路径
        return path

    @staticmethod
    def user(text: str | None=None, images: list[str] | None=None) -> Message:
        parts: list[Part] = []
        if text:
            parts.append(Text(text=text))
        for u in images or []:
            processed_url = msg._process_image_path(u)
            parts.append(Image(url=processed_url))
        return Message(role=Role.user, parts=parts)

    @staticmethod
    def assistant(
        text: str | None = None,
        *,
        thinking: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        reasoning_details: list[dict] | None = None,
    ) -> Message:
        parts: list[Part] = []
        if thinking:
            parts.append(Thinking(text=thinking))
        if text:
            parts.append(Text(text=text))
        return Message(
            role=Role.assistant,
            parts=parts,
            tool_calls=tool_calls,
            reasoning_details=reasoning_details,
        )

    @staticmethod
    def tool(result: str, *, tool_call_id: str, name: str | None = None) -> Message:
        """Create a tool response message.
        
        Args:
            result: The tool execution result.
            tool_call_id: The ID of the tool call this responds to.
            name: The tool name (required for Gemini compatibility via OneAPI/NewAPI).
        """
        return Message(
            role=Role.tool,
            parts=[Text(text=result)],
            tool_call_id=tool_call_id,
            name=name,
        )


class StreamCollector:
    """从流式事件聚合成 Message（用于历史管理）"""

    def __init__(self):
        self._content_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_calls: list[ToolCall] = []
        self._reasoning_details: list[dict] | None = None
        self._usage: Usage | None = None

    def feed(self, event: StreamEvent) -> None:
        """喂入一个流式事件"""
        if event.type == Evt.content and event.delta:
            self._content_parts.append(event.delta)
        elif event.type == Evt.thinking and event.delta:
            self._thinking_parts.append(event.delta)
        elif event.type == Evt.tool_call_start:
            # tool_call_start 只是通知，不聚合
            pass
        elif event.type == Evt.tool_call_delta:
            # tool_call_delta 只是中间状态，不聚合
            pass
        elif event.type == Evt.tool_call_progress:
            # tool_call_progress 只是进度通知，不聚合
            pass
        elif event.type == Evt.tool_call and event.tool_call:
            self._tool_calls.append(event.tool_call)
        elif event.type == Evt.usage and event.usage:
            self._usage = event.usage
        elif event.type == Evt.completed and event.reasoning_details:
            self._reasoning_details = event.reasoning_details

    @property
    def content(self) -> str:
        """聚合的文本内容"""
        return "".join(self._content_parts)

    @property
    def thinking(self) -> str:
        """聚合的思考内容"""
        return "".join(self._thinking_parts)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """工具调用列表"""
        return self._tool_calls

    @property
    def usage(self) -> Usage | None:
        """用量信息"""
        return self._usage

    @property
    def reasoning_details(self) -> list[dict] | None:
        """OpenRouter reasoning_details"""
        return self._reasoning_details

    @property
    def thinking_message(self) -> Message | None:
        """仅包含 thinking 的 Message（无 thinking 返回 None）"""
        if not self._thinking_parts:
            return None
        return Message(
            role=Role.assistant,
            parts=[Thinking(text=self.thinking)],
        )

    @property
    def content_message(self) -> Message | None:
        """仅包含 content 的 Message（无 content 返回 None）"""
        if not self._content_parts:
            return None
        return Message(
            role=Role.assistant,
            parts=[Text(text=self.content)],
            tool_calls=self._tool_calls or None,
            reasoning_details=self._reasoning_details,
        )

    @property
    def message(self) -> Message:
        """聚合为 assistant Message"""
        parts: list[Part] = []
        if self._thinking_parts:
            parts.append(Thinking(text=self.thinking))
        if self._content_parts:
            parts.append(Text(text=self.content))
        return Message(
            role=Role.assistant,
            parts=parts,
            tool_calls=self._tool_calls or None,
            reasoning_details=self._reasoning_details,
        )

    def to_message(self) -> Message:
        """聚合为 assistant Message（别名）"""
        return self.message
