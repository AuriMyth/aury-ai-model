from __future__ import annotations
from typing import AsyncIterator
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from ..types import Message, StreamEvent, Evt, Text, Image, Thinking, ToolCall, Usage
from ..tools import to_openai_tools, normalize_tool_call
from ..instrumentation import set_usage
from ..context import get_ctx
from .base import ProviderAdapter, RequestFeatures, ProviderRoute
from ..errors import (
    ModelTimeoutError, RateLimitError, ModelOverloadedError,
    InvalidRequestError, TransportError,
)

class OpenAIAdapter:
    name = "openai"

    def __init__(self, model: str, base_url: str | None, api_key: str | None, headers: dict | None=None, transport: str | None=None):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.headers = headers or {}
        self.transport = (transport or "chat").lower()
        # Use both sync and async clients
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def route(self, req: RequestFeatures) -> ProviderRoute:
        # 只有在明确指定 transport="responses" 时才使用 responses
        # OpenRouter 等兼容服务优先使用 chat API（更稳定，返回 usage）
        if self.transport == "responses":
            return ProviderRoute(channel="responses", reason="forced_responses")
        return ProviderRoute(channel="chat", reason="chat_default")

    def _to_messages(self, messages: list[Message]) -> list[dict]:
        """映射到 OpenAI ChatCompletions messages，补齐 tool_call_id / tool_calls（用于多轮工具链路）。"""
        out: list[dict] = []
        for m in messages:
            item: dict = {"role": m.role}

            # tool 消息：必须带 tool_call_id，content 为字符串
            if m.role == "tool":
                text = "".join(p.text for p in m.parts if isinstance(p, Text))
                item["content"] = text
                if m.tool_call_id:
                    item["tool_call_id"] = m.tool_call_id
                out.append(item)
                continue

            # assistant 消息：如果包含 tool_calls，需要回传给 API 以配合后续 tool 消息
            if m.role == "assistant" and m.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments_json},
                    }
                    for tc in m.tool_calls
                ]

            # 如果有图片，用 content parts；否则纯文本
            if any(isinstance(p, Image) for p in m.parts):
                parts: list[dict] = []
                for p in m.parts:
                    if isinstance(p, Text):
                        parts.append({"type": "text", "text": p.text})
                    elif isinstance(p, Image):
                        parts.append({"type": "image_url", "image_url": {"url": p.url}})
                    # Thinking 不发送
                item["content"] = parts
            else:
                text = "".join(p.text for p in m.parts if isinstance(p, Text))
                item["content"] = text

            out.append(item)
        return out

    async def ainvoke(self, messages: list[Message], req: RequestFeatures, **kw) -> Message:
        route = self.route(req)
        if route.channel == "responses":
            # Map messages to responses API: use instructions from first system, input from user texts
            sys = next(("".join(p.text for p in m.parts if isinstance(p, Text)) for m in messages if m.role == "system"), None)
            user_text = "\n".join("".join(p.text for p in m.parts if isinstance(p, Text)) for m in messages if m.role in ("user","assistant","tool"))
            payload = {
                "model": self.model,
                "input": user_text or "",
                "extra_headers": {**self.headers, **get_ctx().extra_headers},
            }
            if sys:
                payload["instructions"] = sys
            if tf := kw.get("text_format"):
                payload["text"] = {"format": tf}
            try:
                resp = self.client.responses.create(**payload)
            except Exception as e:
                raise TransportError(str(e)) from e
            # usage
            try:
                u = getattr(resp, "usage", None)
                if u is not None:
                    rt = 0
                    try:
                        rt = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
                    except Exception:
                        rt = 0
                    set_usage(Usage(
                        input_tokens=getattr(u, "input_tokens", 0) or 0,
                        output_tokens=getattr(u, "output_tokens", 0) or 0,
                        reasoning_tokens=rt,
                        total_tokens=getattr(u, "total_tokens", 0) or 0,
                    ))
            except Exception:
                pass
            # content
            text = ""
            try:
                # new responses API often has convenience .output_text
                text = getattr(resp, "output_text", None) or ""
            except Exception:
                text = ""
            if not text:
                try:
                    outs = getattr(resp, "output", []) or []
                    for o in outs:
                        if o.get("type") == "message":
                            for c in o.get("content", []):
                                if c.get("type") in ("output_text","text"):
                                    text += c.get("text", "")
                except Exception:
                    text = ""
            parts = [Text(text=text)] if text else []
            # reasoning summary if available
            try:
                outs = getattr(resp, "output", []) or []
                if req.return_thinking:
                    for o in outs:
                        if o.get("type") == "reasoning" and o.get("summary"):
                            parts.append(Thinking(text=o["summary"][0]["text"]))
                            break
            except Exception:
                pass
            return Message(role="assistant", parts=parts)
        # --- Chat path (default) ---
        payload = dict(
            model=self.model,
            messages=self._to_messages(messages),
            stream=False,
            extra_headers={**self.headers, **get_ctx().extra_headers},
        )
        # Common generation parameters
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.max_completion_tokens is not None:
            payload["max_completion_tokens"] = req.max_completion_tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.stop is not None:
            payload["stop"] = req.stop
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.response_format:
            payload["response_format"] = req.response_format
        if tools := kw.get("tools"):
            payload["tools"] = to_openai_tools(tools, supports_mcp_native=False)
        # Reasoning/thinking support (OpenAI o-series / GPT-5)
        if req.reasoning_effort:
            payload["reasoning_effort"] = req.reasoning_effort
        # Merge user-provided extra_body (provider-specific options)
        if req.extra_body:
            payload.setdefault("extra_body", {}).update(req.extra_body)
        try:
            resp = self.client.chat.completions.create(**payload)
        except Exception as e:
            raise TransportError(str(e)) from e
        # usage (chat)
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                rt = 0
                try:
                    rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
                except Exception:
                    rt = 0
                set_usage(Usage(
                    input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(u, "completion_tokens", 0) or 0,
                    reasoning_tokens=rt,
                    total_tokens=getattr(u, "total_tokens", 0) or 0,
                ))
        except Exception:
            pass
        msg = resp.choices[0].message
        content = msg.content or ""
        parts = [Text(text=content)] if content else []
        # reasoning_content (DeepSeek R1 style)
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning and req.return_thinking:
            parts.append(Thinking(text=reasoning))
        tool_calls = self._extract_tool_calls(msg)
        return Message(role="assistant", parts=parts, tool_calls=tool_calls)

    def _extract_tool_calls(self, msg) -> list[ToolCall] | None:
        """Extract tool calls from response message. Override in subclass for provider-specific handling."""
        raw_calls = getattr(msg, "tool_calls", None)
        if not raw_calls:
            return None
        tool_calls: list[ToolCall] = []
        for tc in raw_calls:
            fn = getattr(tc, "function", None)
            tool_calls.append(normalize_tool_call({
                "id": getattr(tc, "id", None) or "",
                "name": getattr(fn, "name", None) or "",
                "arguments": getattr(fn, "arguments", None) or "{}",
            }))
        return tool_calls or None

    async def astream(self, messages: list[Message], req: RequestFeatures, **kw) -> AsyncIterator[StreamEvent]:
        route = self.route(req)
        if route.channel == "responses":
            # Responses streaming with include_usage if possible
            sys = next(("".join(p.text for p in m.parts if isinstance(p, Text)) for m in messages if m.role == "system"), None)
            user_text = "\n".join("".join(p.text for p in m.parts if isinstance(p, Text)) for m in messages if m.role in ("user","assistant","tool"))
            payload = {
                "model": self.model,
                "input": user_text or "",
                "stream": True,
                "stream_options": {"include_usage": True},
                "extra_headers": {**self.headers, **get_ctx().extra_headers},
            }
            if sys:
                payload["instructions"] = sys
            if tf := kw.get("text_format"):
                payload["text"] = {"format": tf}
            try:
                stream = self.client.responses.stream(**payload)
                # context manager in SDK; emulate basic iteration
                with stream as s:
                    for ev in s:
                        t = getattr(ev, "type", None) or getattr(ev, "event", None) or ""
                        # content delta
                        if isinstance(t, str) and "output_text.delta" in t:
                            delta = getattr(ev, "delta", None) or getattr(getattr(ev, "data", None), "delta", None) or ""
                            if delta:
                                yield StreamEvent(type=Evt.content, delta=str(delta))
                        # reasoning delta
                        if isinstance(t, str) and "reasoning.delta" in t and req.return_thinking:
                            delta = getattr(ev, "delta", None) or getattr(getattr(ev, "data", None), "delta", None) or ""
                            if delta:
                                yield StreamEvent(type=Evt.thinking, delta=str(delta))
                        # usage in completion event
                        if isinstance(t, str) and ("response.completed" in t or "response.summary.delta" in t):
                            try:
                                final = s.get_final_response()
                                u = getattr(final, "usage", None)
                                if u is not None:
                                    rt = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
                                    yield StreamEvent(type=Evt.usage, usage=Usage(
                                        input_tokens=getattr(u, "input_tokens", 0) or 0,
                                        output_tokens=getattr(u, "output_tokens", 0) or 0,
                                        reasoning_tokens=rt,
                                        total_tokens=getattr(u, "total_tokens", 0) or 0,
                                    ))
                            except Exception:
                                pass
                    yield StreamEvent(type=Evt.completed)
            except Exception as e:
                raise TransportError(str(e)) from e
            return
        # --- Chat streaming path (using async client to avoid blocking event loop) ---
        payload = dict(
            model=self.model,
            messages=self._to_messages(messages),
            stream=True,
            extra_headers={**self.headers, **get_ctx().extra_headers},
        )
        # 让兼容 OpenAI 的服务（含 OpenRouter/OneAPI 等）在流式结束时返回 usage（若支持）
        payload["stream_options"] = {"include_usage": True}
        # Common generation parameters
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.max_completion_tokens is not None:
            payload["max_completion_tokens"] = req.max_completion_tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.stop is not None:
            payload["stop"] = req.stop
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.response_format:
            payload["response_format"] = req.response_format
        if tools := kw.get("tools"):
            payload["tools"] = to_openai_tools(tools, supports_mcp_native=False)
        # Reasoning/thinking support (OpenAI o-series / GPT-5)
        if req.reasoning_effort:
            payload["reasoning_effort"] = req.reasoning_effort
        # Merge user-provided extra_body (provider-specific options)
        if req.extra_body:
            payload.setdefault("extra_body", {}).update(req.extra_body)
        try:
            try:
                # Use async client to avoid blocking event loop during streaming
                stream = await self.async_client.chat.completions.create(**payload)
            except Exception:
                # 某些 OpenAI 兼容后端可能不支持 stream_options；兼容性重试一次
                payload.pop("stream_options", None)
                stream = await self.async_client.chat.completions.create(**payload)
            partial_tools: dict[str, dict] = {}
            notified_tools: set[str] = set()  # 记录已通知的工具
            last_progress: dict[str, int] = {}  # 记录上次进度位置
            last_tid: str | None = None
            usage_emitted = False
            # Use async iteration to not block event loop
            async for chunk in stream:
                # 某些实现会在最后一个 chunk 仅带 usage，而没有 choices
                u = getattr(chunk, "usage", None)
                if u is not None and not usage_emitted:
                    rt = 0
                    try:
                        rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
                    except Exception:
                        rt = 0
                    yield StreamEvent(type=Evt.usage, usage=Usage(
                        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                        output_tokens=getattr(u, "completion_tokens", 0) or 0,
                        reasoning_tokens=rt,
                        total_tokens=getattr(u, "total_tokens", 0) or 0,
                    ))
                    usage_emitted = True

                if not getattr(chunk, "choices", None):
                    continue
                ch = getattr(chunk.choices[0], "delta", None)
                if ch is None:
                    continue
                # reasoning_content (DeepSeek R1 style)
                if req.return_thinking:
                    reasoning_delta = getattr(ch, "reasoning_content", None)
                    if reasoning_delta:
                        yield StreamEvent(type=Evt.thinking, delta=reasoning_delta)
                if getattr(ch, "content", None):
                    yield StreamEvent(type=Evt.content, delta=ch.content)
                if getattr(ch, "tool_calls", None):
                    for tc in ch.tool_calls:
                        tid = getattr(tc, "id", None) or last_tid or "_last"
                        if getattr(tc, "id", None):
                            last_tid = tid
                        
                        # ⭐ 1. 首次通知 - tool_call_start
                        if tid not in notified_tools:
                            fn = getattr(tc, "function", None)
                            tool_name = getattr(fn, "name", None) if fn else None
                            
                            if tool_name:
                                # 第一次看到这个工具且有名称，立即通知
                                yield StreamEvent(
                                    type=Evt.tool_call_start,
                                    tool_call=ToolCall(
                                        id=tid,
                                        name=tool_name,
                                        arguments_json="",
                                    )
                                )
                                notified_tools.add(tid)
                                last_progress[tid] = 0
                        
                        entry = partial_tools.setdefault(tid, {"id": tid, "name": "", "arguments": ""})
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                entry["name"] += fn.name
                            # arguments 可能是空字符串，用 is not None 而不是 truthy 检查
                            args_delta = getattr(fn, "arguments", None)
                            if args_delta is not None:
                                # ⭐ 2. 参数增量 - tool_call_delta
                                if args_delta:  # 非空才发送
                                    yield StreamEvent(
                                        type=Evt.tool_call_delta,
                                        tool_call_delta={
                                            "call_id": tid,
                                            "arguments_delta": args_delta,
                                        }
                                    )
                                
                                entry["arguments"] += args_delta
                                
                                # ⭐ 3. 进度通知 - tool_call_progress
                                current_size = len(entry["arguments"])
                                prev_size = last_progress.get(tid, 0)
                                
                                # 每1KB发送一次进度
                                if current_size - prev_size >= 1024:
                                    yield StreamEvent(
                                        type=Evt.tool_call_progress,
                                        tool_call_progress={
                                            "call_id": tid,
                                            "bytes_received": current_size,
                                            "last_delta_size": current_size - prev_size,
                                        }
                                    )
                                    last_progress[tid] = current_size
            for _, v in partial_tools.items():
                yield StreamEvent(type=Evt.tool_call, tool_call=normalize_tool_call(v))
            yield StreamEvent(type=Evt.completed)
        except Exception as e:
            raise TransportError(str(e)) from e
