from __future__ import annotations
from typing import AsyncIterator
from .openai import OpenAIAdapter
from .base import RequestFeatures
from ..types import Message, StreamEvent, Evt, Text, Image, Thinking, ToolCall, Usage
from ..tools import normalize_tool_call, to_openai_tools
from ..instrumentation import set_usage
from ..errors import TransportError
from ..context import get_ctx


class OpenRouterAdapter(OpenAIAdapter):
    """OpenRouter adapter with support for:
    - Unified reasoning interface (reasoning.effort, reasoning.summary)
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
        
        # OpenRouter 统一的 reasoning 接口
        # 参考: https://openrouter.ai/docs/api-reference
        if req.return_thinking or req.reasoning_effort:
            reasoning_obj = {}
            
            # effort: "low" | "medium" | "high" | "max" | "auto" | null
            if req.reasoning_effort:
                reasoning_obj["effort"] = req.reasoning_effort
            elif req.return_thinking:
                # 如果只设置了 return_thinking，使用默认 effort
                reasoning_obj["effort"] = "medium"
            
            # summary: "auto" | "concise" | "detailed" | null
            # 暂不设置，使用模型默认值
            
            if reasoning_obj:
                extra["reasoning"] = reasoning_obj
        
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

    def _parse_xml_tool_calls(self, content: str) -> tuple[str, list[ToolCall] | None]:
        """Parse XML-style tool calls from content (DeepSeek/Zhipu/Qwen format).
        
        Returns:
            tuple of (cleaned_content, tool_calls)
        """
        import re
        import json as _json
        import uuid
        
        tool_calls: list[ToolCall] = []
        
        # Pattern for <tool_call>{"name": ..., "arguments": ...}</tool_call>
        tool_call_pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
        
        # Pattern for <function_calls><invoke name="..." ...>...</invoke></function_calls>
        function_calls_pattern = re.compile(
            r'<function_calls>.*?<invoke\s+name=["\']([^"\']+)["\'].*?>\s*(.*?)\s*</invoke>.*?</function_calls>',
            re.DOTALL
        )
        
        # Try tool_call format first
        for match in tool_call_pattern.finditer(content):
            try:
                tc_json = _json.loads(match.group(1))
                name = tc_json.get("name", "")
                arguments = tc_json.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = _json.dumps(arguments)
                tool_calls.append(ToolCall(
                    id=f"xml_tc_{uuid.uuid4().hex[:8]}",
                    name=name,
                    arguments_json=arguments,
                ))
            except Exception:
                pass
        
        # Try function_calls format
        for match in function_calls_pattern.finditer(content):
            try:
                name = match.group(1)
                # Arguments might be in various formats
                args_str = match.group(2).strip()
                try:
                    arguments = _json.loads(args_str) if args_str else {}
                    if isinstance(arguments, dict):
                        arguments = _json.dumps(arguments)
                    else:
                        arguments = args_str
                except Exception:
                    arguments = args_str
                tool_calls.append(ToolCall(
                    id=f"xml_fc_{uuid.uuid4().hex[:8]}",
                    name=name,
                    arguments_json=arguments if isinstance(arguments, str) else _json.dumps(arguments),
                ))
            except Exception:
                pass
        
        if tool_calls:
            # Remove XML tool calls from content
            cleaned = tool_call_pattern.sub('', content)
            cleaned = function_calls_pattern.sub('', cleaned)
            # Also remove incomplete patterns
            cleaned = re.sub(r'<tool_call>.*', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'<function_calls>.*', '', cleaned, flags=re.DOTALL)
            return cleaned.strip(), tool_calls
        
        return content, None

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

        # OpenRouter reasoning 接口：通过 extra_body.reasoning 传递
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
        
        # If no native tool_calls, try parsing XML-style tool calls from content (DeepSeek/Zhipu/Qwen)
        if not tool_calls and content and ("<tool_call>" in content or "<function_calls>" in content):
            cleaned_content, xml_tool_calls = self._parse_xml_tool_calls(content)
            if xml_tool_calls:
                tool_calls = xml_tool_calls
                # Update parts with cleaned content
                parts = [Text(text=cleaned_content)] if cleaned_content else []
        
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

        # OpenRouter reasoning 接口：通过 extra_body.reasoning 传递
        extra_body = self._build_extra_body(req, req.extra_body)
        if extra_body:
            payload["extra_body"] = extra_body

        try:
            try:
                # Use async client to avoid blocking event loop during streaming
                stream = await self.async_client.chat.completions.create(**payload)
            except Exception:
                payload.pop("stream_options", None)
                stream = await self.async_client.chat.completions.create(**payload)

            partial_tools: dict[str, dict] = {}
            last_tid: str | None = None
            usage_emitted = False
            # Accumulate reasoning_details from streaming chunks
            accumulated_reasoning_details: list[dict] = []
            # Accumulate content for XML tool call detection
            accumulated_content: list[str] = []
            # Track which tools have been notified via tool_call_start
            notified_tools: set[str] = set()
            last_progress: dict[str, int] = {}

            # Use async iteration to not block event loop
            async for chunk in stream:
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
                    accumulated_content.append(ch.content)
                    yield StreamEvent(type=Evt.content, delta=ch.content)

                if getattr(ch, "tool_calls", None):
                    for tc in ch.tool_calls:
                        # OpenAI 流式格式
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
                            # 注意：arguments 可能是空字符串，所以用 is not None 而不是 truthy 检查
                            args_delta = getattr(fn, "arguments", None)
                            if args_delta is not None:
                                entry["arguments"] += args_delta
                        
                        # Emit tool_call_start when we first see a tool with id and name
                        tool_id = entry["id"]
                        tool_name = entry["name"]
                        if tool_id and tool_name and tool_id not in notified_tools:
                            yield StreamEvent(
                                type=Evt.tool_call_start,
                                tool_call=ToolCall(
                                    id=tool_id,
                                    name=tool_name,
                                    arguments_json="",
                                )
                            )
                            notified_tools.add(tool_id)
                            last_progress[tool_id] = 0
                        
                        # Emit tool_call_progress for arguments streaming
                        if tool_id and tool_id in notified_tools:
                            current_len = len(entry["arguments"])
                            if current_len > last_progress.get(tool_id, 0):
                                yield StreamEvent(
                                    type=Evt.tool_call_progress,
                                    tool_call_progress={
                                        "call_id": tool_id,
                                        "bytes_received": current_len,
                                    }
                                )
                                last_progress[tool_id] = current_len

            # Emit accumulated tool calls
            for _, v in partial_tools.items():
                normalized = normalize_tool_call(v)
                yield StreamEvent(type=Evt.tool_call, tool_call=normalized)
            
            # If no native tool_calls, check for XML-style tool calls in accumulated content
            if not partial_tools:
                full_content = "".join(accumulated_content)
                if "<tool_call>" in full_content or "<function_calls>" in full_content:
                    _, xml_tool_calls = self._parse_xml_tool_calls(full_content)
                    if xml_tool_calls:
                        for tc in xml_tool_calls:
                            yield StreamEvent(type=Evt.tool_call, tool_call=tc)

            # Emit completed event with reasoning_details for Gemini 3 / DeepSeek tool calling
            yield StreamEvent(
                type=Evt.completed,
                reasoning_details=accumulated_reasoning_details or None
            )
        except Exception as e:
            raise TransportError(str(e)) from e
