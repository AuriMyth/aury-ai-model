from __future__ import annotations
from logging import DEBUG
from typing import AsyncIterator
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI, Timeout as OpenAITimeout
from ..types import Message, StreamEvent, Evt, Text, Image, Thinking, ToolCall, Usage
from ..tools import to_openai_tools, normalize_tool_call
from ..instrumentation import set_usage
from ..context import get_ctx
from .base import ProviderAdapter, RequestFeatures, ProviderRoute
from ..errors import (
    ModelTimeoutError, RateLimitError, ModelOverloadedError,
    InvalidRequestError, TransportError,
)
from .. import logger, TRACE

# Default timeout: 30s connect, 10min read (for long streaming responses)
DEFAULT_TIMEOUT = OpenAITimeout(connect=30.0, read=600.0, write=60.0, pool=30.0)


class OpenAIAdapter:
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None,
        api_key: str | None,
        headers: dict | None = None,
        transport: str | None = None,
        timeout: float | OpenAITimeout | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.headers = headers or {}
        self.transport = (transport or "chat").lower()
        
        # Build timeout config
        if timeout is None:
            client_timeout = DEFAULT_TIMEOUT
        elif isinstance(timeout, (int, float)):
            # Single value: use as read timeout, keep defaults for others
            client_timeout = OpenAITimeout(connect=30.0, read=float(timeout), write=60.0, pool=30.0)
        else:
            client_timeout = timeout
        
        # Use both sync and async clients with timeout
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=client_timeout)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=client_timeout)

    def route(self, req: RequestFeatures) -> ProviderRoute:
        # 只有在明确指定 transport="responses" 时才使用 responses
        # OpenRouter 等兼容服务优先使用 chat API（更稳定，返回 usage）
        if self.transport == "responses":
            return ProviderRoute(channel="responses", reason="forced_responses")
        return ProviderRoute(channel="chat", reason="chat_default")

    def _to_messages(self, messages: list[Message]) -> list[dict]:
        """映射到 OpenAI ChatCompletions messages。
        
        Claude 模型（带 thinking）使用 Anthropic/Bedrock 格式：
        - content 数组格式包含 thinking blocks
        - thinking 必须在 content 数组最前面
        - tool_use 在 thinking 之后
        - tool_result 使用 role="user" + tool_result block
        
        其他模型使用标准 OpenAI 格式：
        - content 为字符串
        - tool_calls 为独立字段
        - 完全忽略 Thinking blocks
        """
        import json as _json
        
        is_claude = "claude" in self.model.lower()
        # Claude 且消息中有 thinking 时使用 Anthropic 格式
        use_anthropic_format = is_claude and any(
            any(isinstance(p, Thinking) for p in m.parts)
            for m in messages
        )
        
        if use_anthropic_format:
            return self._to_messages_anthropic(messages)
        else:
            return self._to_messages_openai(messages)
    
    def _to_messages_anthropic(self, messages: list[Message]) -> list[dict]:
        """Claude/Bedrock 格式：thinking + tool_use 在 content 数组中。"""
        import json as _json
        out: list[dict] = []
        
        for m in messages:
            item: dict = {"role": m.role}
            
            # tool 消息 -> 转为普通 user 消息
            if m.role == "tool":
                text = "".join(p.text for p in m.parts if isinstance(p, Text))
                item["role"] = "user"
                item["content"] = [{"type": "text", "text": f"[工具返回结果]\n{text}"}]
                out.append(item)
                continue
            
            # 构建 content 数组
            parts: list[dict] = []
            
            # 1. thinking 必须在最前面
            for p in m.parts:
                if isinstance(p, Thinking) and p.text:
                    parts.append({"type": "thinking", "thinking": p.text})
            
            # 2. text 内容
            for p in m.parts:
                if isinstance(p, Text) and p.text:
                    parts.append({"type": "text", "text": p.text})
            
            # 3. 图片
            for p in m.parts:
                if isinstance(p, Image):
                    parts.append({"type": "image_url", "image_url": {"url": p.url}})
            
            # 4. assistant 消息：tool_use 在最后
            if m.role == "assistant" and m.tool_calls:
                # 确保有 text（Bedrock 要求）
                if not any(isinstance(p, Text) and p.text for p in m.parts):
                    parts.append({"type": "text", "text": "调用工具"})
                
                for tc in m.tool_calls:
                    try:
                        args = _json.loads(tc.arguments_json or "{}")
                    except Exception:
                        args = {}
                    parts.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": args,
                    })
            
            # 5. Bedrock 要求：assistant 消息如果只有 thinking（无 text/tool_use），需要添加占位文本
            # 否则会报错：all messages must have non-empty content except for the optional final assistant message
            if m.role == "assistant" and parts:
                has_text_or_tool = any(
                    p.get("type") in ("text", "tool_use") for p in parts
                )
                if not has_text_or_tool:
                    parts.append({"type": "text", "text": "(思考中)"})
            
            item["content"] = parts if parts else None
            out.append(item)
        
        return out
    
    def _to_messages_openai(self, messages: list[Message]) -> list[dict]:
        """标准 OpenAI 格式：完全忽略 Thinking blocks。"""
        out: list[dict] = []
        
        for m in messages:
            item: dict = {"role": m.role}
            has_images = any(isinstance(p, Image) for p in m.parts)
            
            # tool 消息
            if m.role == "tool":
                text = "".join(p.text for p in m.parts if isinstance(p, Text))
                item["content"] = text
                if m.tool_call_id:
                    item["tool_call_id"] = m.tool_call_id
                # name is required for Gemini compatibility via OneAPI/NewAPI
                # fallback to "_tool_response_" if not provided (avoid "unknown" as Gemini may treat it as a tool name)
                item["name"] = m.name or "_tool_response_"
                out.append(item)
                continue
            
            # assistant 消息的 tool_calls
            if m.role == "assistant" and m.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments_json},
                    }
                    for tc in m.tool_calls
                ]
            
            # content 处理
            if has_images:
                # 图片用数组格式（只包含 text 和 image，忽略 Thinking）
                parts: list[dict] = []
                for p in m.parts:
                    if isinstance(p, Text) and p.text:
                        parts.append({"type": "text", "text": p.text})
                    elif isinstance(p, Image):
                        parts.append({"type": "image_url", "image_url": {"url": p.url}})
                item["content"] = parts if parts else None
            else:
                # 纯文本（忽略 Thinking）
                text = "".join(p.text for p in m.parts if isinstance(p, Text))
                item["content"] = text or None
            
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
                resp = await self.async_client.responses.create(**payload)
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
            payload["tools"] = to_openai_tools(tools, supports_mcp_native=False, model=self.model)
        # Only for Claude models (claude-* or anthropic provider)
        is_claude = "claude" in self.model.lower()
        # Reasoning/thinking support
        # For Claude via NewAPI: use reasoning_effort which maps to thinking mode
        # For OpenAI o-series: use reasoning_effort directly
        if req.return_thinking and is_claude:
            # NewAPI uses reasoning_effort for Claude thinking mode
            payload["reasoning_effort"] = req.reasoning_effort or "medium"
        elif req.reasoning_effort:
            payload["reasoning_effort"] = req.reasoning_effort
        # Merge user-provided extra_body (provider-specific options, can override above)
        if req.extra_body:
            payload.setdefault("extra_body", {}).update(req.extra_body)
        try:
            resp = await self.async_client.chat.completions.create(**payload)
        except Exception as e:
            raise TransportError(str(e)) from e
        logger.debug("[ainvoke] model=%s resp=%s", self.model, resp)
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
                # Use async client to avoid blocking event loop
                stream = await self.async_client.responses.stream(**payload)
                # Use async iteration
                async for ev in stream:
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
                            # Note: In async streaming, we might not have get_final_response()
                            # Extract usage from the event itself if available
                            u = getattr(ev, "usage", None)
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
        converted_messages = self._to_messages(messages)
        logger.debug("[astream] model=%s messages=%s", self.model, converted_messages)
        
        payload = dict(
            model=self.model,
            messages=converted_messages,
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
            payload["tools"] = to_openai_tools(tools, supports_mcp_native=False, model=self.model)
            # Debug: log full tools payload for troubleshooting
            import json
            logger.debug("[astream] tools payload: %s", json.dumps(payload["tools"], ensure_ascii=False, indent=2))
        # Only for Claude models (claude-* or anthropic provider)
        is_claude = "claude" in self.model.lower()
        # Reasoning/thinking support
        # For Claude via NewAPI: use reasoning_effort which maps to thinking mode
        # For OpenAI o-series: use reasoning_effort directly
        if req.return_thinking and is_claude:
            # NewAPI uses reasoning_effort for Claude thinking mode
            payload["reasoning_effort"] = req.reasoning_effort or "medium"
        elif req.reasoning_effort:
            payload["reasoning_effort"] = req.reasoning_effort
        # Merge user-provided extra_body (provider-specific options, can override above)
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
            chunk_count = 0
            # Use async iteration to not block event loop
            async for chunk in stream:
                chunk_count += 1
                # TRACE 级别 + 采样：首个 chunk 和每 50 个 chunk 记录一次
                if chunk_count == 1 or chunk_count % 50 == 0:
                    logger.log(DEBUG, "[astream] chunk #%d: %s", chunk_count, chunk)
                # 某些实现会在最后一个 chunk 仅带 usage，而没有 choices
                u = getattr(chunk, "usage", None)
                if u is not None and not usage_emitted:
                    rt = 0
                    cache_read = 0
                    cache_write = 0
                    try:
                        # reasoning_tokens from completion_tokens_details
                        rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
                    except Exception:
                        pass
                    try:
                        # cache tokens from prompt_tokens_details
                        ptd = getattr(u, "prompt_tokens_details", None)
                        if ptd:
                            cache_read = getattr(ptd, "cached_tokens", 0) or 0
                        # Claude via NewAPI: claude_cache_* fields
                        if cache_read == 0:
                            cache_read = getattr(u, "claude_cache_read_tokens", 0) or 0
                        cache_write = getattr(u, "claude_cache_creation_5_m_tokens", 0) or 0
                        cache_write += getattr(u, "claude_cache_creation_1_h_tokens", 0) or 0
                    except Exception:
                        pass
                    yield StreamEvent(type=Evt.usage, usage=Usage(
                        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                        output_tokens=getattr(u, "completion_tokens", 0) or 0,
                        reasoning_tokens=rt,
                        cache_read_tokens=cache_read,
                        cache_write_tokens=cache_write,
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
                    # Claude thinking (may be in 'thinking' attribute)
                    thinking_delta = getattr(ch, "thinking", None)
                    if thinking_delta:
                        yield StreamEvent(type=Evt.thinking, delta=thinking_delta)
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
