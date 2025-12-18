from __future__ import annotations
from typing import AsyncIterator
from .openai import OpenAIAdapter
from .base import RequestFeatures
from ..types import Message, StreamEvent, Evt, Text, Thinking, ToolCall, Usage
from ..tools import normalize_tool_call, to_openai_tools
from ..instrumentation import set_usage
from ..errors import TransportError
from ..context import get_ctx


class OpenRouterAdapter(OpenAIAdapter):
    """OpenRouter adapter with support for:
    - OpenRouter-specific reasoning parameters (include_reasoning, reasoning.effort)
    - Gemini 3 thoughtSignature preservation for function calling
    - Multiple reasoning field formats (reasoning, reasoning_content)
    """
    name = "openrouter"

    def __init__(self, model: str, base_url: str | None, api_key: str | None, headers: dict | None = None):
        super().__init__(
            model=model,
            base_url=base_url or "https://openrouter.ai/api/v1",
            api_key=api_key,
            headers=headers,
        )

    def _to_messages(self, messages: list[Message]) -> list[dict]:
        """Override to preserve reasoning_details for OpenRouter tool calling."""
        from ..types import Image
        out: list[dict] = []
        for m in messages:
            item: dict = {"role": m.role}

            # tool 消息
            if m.role == "tool":
                text = "".join(p.text for p in m.parts if isinstance(p, Text))
                item["content"] = text
                if m.tool_call_id:
                    item["tool_call_id"] = m.tool_call_id
                out.append(item)
                continue

            # assistant 消息
            if m.role == "assistant":
                # tool_calls
                if m.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments_json},
                        }
                        for tc in m.tool_calls
                    ]
                # OpenRouter: reasoning_details must be passed back unmodified for tool calling
                if m.reasoning_details:
                    item["reasoning_details"] = m.reasoning_details

            # content
            if any(isinstance(p, Image) for p in m.parts):
                parts: list[dict] = []
                for p in m.parts:
                    if isinstance(p, Text):
                        parts.append({"type": "text", "text": p.text})
                    elif isinstance(p, Image):
                        parts.append({"type": "image_url", "image_url": {"url": p.url}})
                item["content"] = parts
            else:
                text = "".join(p.text for p in m.parts if isinstance(p, Text))
                item["content"] = text

            out.append(item)
        return out

    def _build_extra_body(self, req: RequestFeatures, existing: dict | None = None) -> dict:
        """Build extra_body with OpenRouter-specific parameters."""
        extra = existing.copy() if existing else {}
        # OpenRouter uses include_reasoning + reasoning.effort for thinking models
        if req.return_thinking:
            extra["include_reasoning"] = True
            effort = req.reasoning_effort or "medium"
            extra["reasoning"] = {"effort": effort}
        return extra

    def _extract_tool_calls_from_raw(self, raw_tool_calls: list[dict] | None) -> list[ToolCall] | None:
        """Extract tool calls from raw JSON response."""
        if not raw_tool_calls:
            return None
        tool_calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            fn = tc.get("function") or {}
            tool_calls.append(normalize_tool_call({
                "id": tc.get("id") or "",
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "{}",
            }))
        return tool_calls or None

    async def ainvoke(self, messages: list[Message], req: RequestFeatures, **kw) -> Message:
        """Override to use OpenRouter-specific reasoning parameters and extract thoughtSignature."""
        import json as _json
        route = self.route(req)
        if route.channel == "responses":
            return await super().ainvoke(messages, req, **kw)

        # --- Chat path with OpenRouter extensions ---
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

        # OpenRouter-specific: reasoning parameters go in extra_body
        extra_body = self._build_extra_body(req, req.extra_body)
        if extra_body:
            payload["extra_body"] = extra_body

        try:
            # Use with_raw_response to get the original JSON (SDK doesn't expose reasoning_details)
            raw_resp = self.client.chat.completions.with_raw_response.create(**payload)
            resp = raw_resp.parse()
            # Parse raw JSON to extract reasoning_details and thoughtSignature
            raw_json: dict = {}
            try:
                raw_json = _json.loads(raw_resp.text)
            except Exception:
                pass
        except Exception as e:
            raise TransportError(str(e)) from e

        # usage
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
        
        # Extract images (OpenRouter-specific: choices[0].message.images)
        try:
            images_field = None
            try:
                # Prefer raw_json because SDK may not expose .images
                choices = raw_json.get("choices") or []
                if choices:
                    raw_msg = choices[0].get("message") or {}
                    images_field = raw_msg.get("images")
            except Exception:
                images_field = None
            if images_field and isinstance(images_field, list):
                for img in images_field:
                    # Expected shapes: {"image_url": {"url": "data:..."}} or {"image_url": {"url": "https://..."}}
                    if isinstance(img, dict):
                        iu = img.get("image_url") or {}
                        url = iu.get("url") or None
                        if url:
                            parts.append(Text(text=""))  # keep position stable if content exists
                            parts.append(Image(url=url))
        except Exception:
            pass

        # OpenRouter uses 'reasoning' field, DeepSeek uses 'reasoning_content'
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        if reasoning and req.return_thinking:
            parts.append(Thinking(text=reasoning))

        # Extract tool_calls and reasoning_details from raw JSON
        raw_tool_calls = None
        reasoning_details = None
        try:
            choices = raw_json.get("choices") or []
            if choices:
                raw_msg = choices[0].get("message") or {}
                raw_tool_calls = raw_msg.get("tool_calls")
                reasoning_details = raw_msg.get("reasoning_details")
        except Exception:
            pass

        tool_calls = self._extract_tool_calls_from_raw(raw_tool_calls) if raw_tool_calls else None
        
        # Return Message with reasoning_details preserved for tool calling
        return Message(role="assistant", parts=parts, tool_calls=tool_calls, reasoning_details=reasoning_details)

    async def astream(self, messages: list[Message], req: RequestFeatures, **kw) -> AsyncIterator[StreamEvent]:
        """Override to handle OpenRouter-specific streaming with reasoning_details."""
        route = self.route(req)
        if route.channel == "responses":
            async for event in super().astream(messages, req, **kw):
                yield event
            return

        # --- Chat streaming path with OpenRouter extensions ---
        payload = dict(
            model=self.model,
            messages=self._to_messages(messages),
            stream=True,
            extra_headers={**self.headers, **get_ctx().extra_headers},
        )
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

        # OpenRouter-specific: reasoning parameters go in extra_body
        extra_body = self._build_extra_body(req, req.extra_body)
        if extra_body:
            payload["extra_body"] = extra_body

        try:
            try:
                stream = self.client.chat.completions.create(**payload)
            except Exception:
                payload.pop("stream_options", None)
                stream = self.client.chat.completions.create(**payload)

            partial_tools: dict[str, dict] = {}
            last_tid: str | None = None
            usage_emitted = False
            # Accumulate reasoning_details from streaming chunks
            accumulated_reasoning_details: list[dict] = []

            for chunk in stream:
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

                # Extract reasoning_details from streaming chunk (for Gemini 3 / DeepSeek tool calling)
                try:
                    chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else {}
                    choices = chunk_dict.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        rd = delta.get("reasoning_details")
                        if rd and isinstance(rd, list):
                            accumulated_reasoning_details.extend(rd)
                except Exception:
                    pass

                # OpenRouter uses 'reasoning' field, DeepSeek uses 'reasoning_content'
                if req.return_thinking:
                    reasoning_delta = getattr(ch, "reasoning", None) or getattr(ch, "reasoning_content", None)
                    if reasoning_delta:
                        yield StreamEvent(type=Evt.thinking, delta=reasoning_delta)

                if getattr(ch, "content", None):
                    yield StreamEvent(type=Evt.content, delta=ch.content)

                if getattr(ch, "tool_calls", None):
                    for tc in ch.tool_calls:
                        # OpenAI 流式格式：第一个 chunk 带 id+name+index，后续 chunks 只带 index+arguments
                        # 所以必须用 index 作为主 key
                        idx = getattr(tc, "index", None)
                        tid = getattr(tc, "id", None)
                        
                        # 优先用 index 作为 key（最可靠），否则用 id
                        key = f"_idx_{idx}" if idx is not None else (tid or "_last")
                        entry = partial_tools.setdefault(key, {"id": "", "name": "", "arguments": ""})
                        
                        # 更新 id（只在第一个 chunk 出现）
                        if tid:
                            entry["id"] = tid
                        
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            # name 只在第一个 chunk 出现
                            if getattr(fn, "name", None):
                                entry["name"] = fn.name
                            # arguments 可能分多个 chunk，需要累加
                            if getattr(fn, "arguments", None):
                                entry["arguments"] += fn.arguments

            # Emit accumulated tool calls
            for _, v in partial_tools.items():
                normalized = normalize_tool_call(v)
                yield StreamEvent(type=Evt.tool_call, tool_call=normalized)

            # Emit completed event with reasoning_details for Gemini 3 / DeepSeek tool calling
            yield StreamEvent(
                type=Evt.completed,
                reasoning_details=accumulated_reasoning_details or None
            )
        except Exception as e:
            raise TransportError(str(e)) from e
