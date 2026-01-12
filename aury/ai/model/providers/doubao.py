from __future__ import annotations
from typing import AsyncIterator, Any
import importlib
from ..types import Message, StreamEvent, Evt, Text, Image, Thinking, ToolCall, Usage
from .base import RequestFeatures, ProviderRoute
from ..errors import ProviderNotInstalledError, TransportError
from ..instrumentation import set_usage
from ..tools import to_openai_tools, normalize_tool_call
from ..context import get_ctx

class DoubaoArkAdapter:
    name = "doubao"

    def __init__(self, model: str, base_url: str | None, api_key: str | None, headers: dict | None=None):
        spec = importlib.util.find_spec("volcenginesdkarkruntime")
        if spec is None:
            raise ProviderNotInstalledError("Install VolcEngine SDK: pip install 'volcengine-python-sdk[ark]'")
        self._ark = importlib.import_module("volcenginesdkarkruntime")
        # 只支持 AsyncArk（你的包没有同步实现；同时避免阻塞事件循环）
        async_cls = getattr(self._ark, "AsyncArk", None)
        if async_cls is None:
            raise ProviderNotInstalledError(
                "VolcEngine SDK missing AsyncArk. Please upgrade: pip install -U 'volcengine-python-sdk[ark]'"
            )
        self.client = async_cls(base_url=base_url or "https://ark.cn-beijing.volces.com/api/v3", api_key=api_key)
        self.model = model
        self.headers = headers or {}

    def route(self, req: RequestFeatures) -> ProviderRoute:
        # Doubao/Ark: 使用 ChatCompletions（与文档一致，工具/多模态更兼容）
        return ProviderRoute(channel="chat", reason="ark_chat_default")

    def _to_messages(self, messages: list[Message]) -> list[dict]:
        # ChatCompletions: 若有图片则 content 为 parts；否则为纯文本字符串
        out: list[dict] = []
        for m in messages:
            # tool 消息：必须带 tool_call_id，content 为字符串
            if m.role == "tool":
                text = "".join(p.text for p in m.parts if isinstance(p, Text))
                item: dict = {"role": "tool", "content": text}
                if m.tool_call_id:
                    item["tool_call_id"] = m.tool_call_id
                out.append(item)
                continue

            if any(isinstance(p, Image) for p in m.parts):
                parts: list[dict] = []
                for p in m.parts:
                    if isinstance(p, Text):
                        parts.append({"type": "text", "text": p.text})
                    elif isinstance(p, Image):
                        parts.append({"type": "image_url", "image_url": {"url": p.url}})
                item: dict = {"role": m.role, "content": parts}
                # assistant tool_calls（用于多轮工具链路）
                if m.role == "assistant" and m.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments_json},
                        }
                        for tc in m.tool_calls
                    ]
                out.append(item)
            else:
                text = "".join(p.text for p in m.parts if isinstance(p, Text))
                item: dict = {"role": m.role, "content": text}
                if m.role == "assistant" and m.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments_json},
                        }
                        for tc in m.tool_calls
                    ]
                out.append(item)
        return out

    def _extract_tool_calls(self, msg) -> list[ToolCall] | None:
        raw_calls = getattr(msg, "tool_calls", None)
        if not raw_calls:
            return None
        out: list[ToolCall] = []
        for tc in raw_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) if fn is not None else None
            args = getattr(fn, "arguments", None) if fn is not None else None
            out.append(normalize_tool_call({
                "id": getattr(tc, "id", None) or "",
                "name": name or "",
                "arguments": args or "{}",
            }))
        return out or None

    def _to_usage(self, u: Any) -> Usage:
        """兼容 OpenAI/Ark 的 usage 表达（对象或 dict），并尽量提取 reasoning_tokens。"""
        if u is None:
            return Usage()
        # dict style
        if isinstance(u, dict):
            details = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
            rt = 0
            if isinstance(details, dict):
                rt = details.get("reasoning_tokens", 0) or 0
            return Usage(
                input_tokens=u.get("prompt_tokens", 0) or u.get("input_tokens", 0) or 0,
                output_tokens=u.get("completion_tokens", 0) or u.get("output_tokens", 0) or 0,
                reasoning_tokens=rt,
                total_tokens=u.get("total_tokens", 0) or 0,
            )
        # object style
        rt = 0
        try:
            det = getattr(u, "completion_tokens_details", None) or getattr(u, "output_tokens_details", None)
            if det is not None:
                if isinstance(det, dict):
                    rt = det.get("reasoning_tokens", 0) or 0
                else:
                    rt = getattr(det, "reasoning_tokens", 0) or 0
        except Exception:
            rt = 0
        return Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or getattr(u, "output_tokens", 0) or 0,
            reasoning_tokens=rt,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
        )

    async def ainvoke(self, messages: list[Message], req: RequestFeatures, **kw) -> Message:
        payload: dict = {
            "model": self.model,
            "messages": self._to_messages(messages),
            "stream": False,
            "extra_headers": {**self.headers, **get_ctx().extra_headers},
        }
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
        # Reasoning/thinking support
        payload["thinking"] = {"type": "enabled" if req.return_thinking else "disabled"}
        if req.reasoning_effort:
            payload["reasoning_effort"] = req.reasoning_effort
        # Tools
        if tools := kw.get("tools"):
            payload["tools"] = to_openai_tools(tools)
        # Provider-specific extra_body
        if req.extra_body:
            payload.update(req.extra_body)
        try:
            resp = await self.client.chat.completions.create(**payload)
        except Exception as e:
            raise TransportError(str(e)) from e

        # usage（OpenAI 兼容：prompt_tokens/completion_tokens/total_tokens）
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                set_usage(self._to_usage(u))
        except Exception:
            pass

        m = resp.choices[0].message
        content = getattr(m, "content", None) or ""
        parts = [Text(text=content)] if content else []
        # reasoning_content（文档提到的输出字段）
        reasoning = getattr(m, "reasoning_content", None)
        if reasoning and req.return_thinking:
            parts.append(Thinking(text=str(reasoning)))

        tool_calls = self._extract_tool_calls(m)
        return Message(role="assistant", parts=parts, tool_calls=tool_calls)

    async def astream(self, messages: list[Message], req: RequestFeatures, **kw) -> AsyncIterator[StreamEvent]:
        payload: dict = {
            "model": self.model,
            "messages": self._to_messages(messages),
            "stream": True,
            "extra_headers": {**self.headers, **get_ctx().extra_headers},
        }
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
        # Reasoning/thinking support
        payload["thinking"] = {"type": "enabled" if req.return_thinking else "disabled"}
        if req.reasoning_effort:
            payload["reasoning_effort"] = req.reasoning_effort
        # Tools
        if tools := kw.get("tools"):
            payload["tools"] = to_openai_tools(tools)
        # Provider-specific extra_body
        if req.extra_body:
            payload.update(req.extra_body)
        try:
            # 尝试开启流式 usage（若后端/SDK 不支持则自动回退）
            payload["stream_options"] = {"include_usage": True}
            try:
                stream = await self.client.chat.completions.create(**payload)
            except Exception:
                payload.pop("stream_options", None)
                stream = await self.client.chat.completions.create(**payload)

            partial_tools: dict[str, dict] = {}
            notified_tools: set[str] = set()  # 记录已通知的工具
            last_progress: dict[str, int] = {}  # 记录上次进度位置
            usage_emitted = False
            last_tid: str | None = None
            async for chunk in stream:
                # Ark/OpenAI 兼容 SDK：优先用 model_dump() 解析，避免属性缺失导致 usage 丢失
                d = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                if isinstance(d, dict):
                    u = d.get("usage")
                    if u is not None and not usage_emitted:
                        usage_emitted = True
                        uu = self._to_usage(u)
                        set_usage(uu)
                        yield StreamEvent(type=Evt.usage, usage=uu)
                    choices = d.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0] or {}).get("delta") or {}
                    if delta.get("content"):
                        yield StreamEvent(type=Evt.content, delta=delta["content"])
                    if req.return_thinking and delta.get("reasoning_content"):
                        yield StreamEvent(type=Evt.thinking, delta=delta["reasoning_content"])
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            fn = tc.get("function") or {}
                            tid = tc.get("id") or last_tid or "_last"
                            if tc.get("id"):
                                last_tid = tid
                            
                            # ⭐ 首次通知 - tool_call_start
                            if tid not in notified_tools:
                                tool_name = fn.get("name")
                                if tool_name:
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
                            if fn.get("name"):
                                entry["name"] += fn["name"]
                            args_delta = fn.get("arguments")
                            if args_delta:
                                # ⭐ 参数增量 - tool_call_delta
                                yield StreamEvent(
                                    type=Evt.tool_call_delta,
                                    tool_call_delta={
                                        "call_id": tid,
                                        "arguments_delta": args_delta,
                                    }
                                )
                                
                                entry["arguments"] += args_delta
                                
                                # ⭐ 进度通知 - tool_call_progress
                                current_size = len(entry["arguments"])
                                prev_size = last_progress.get(tid, 0)
                                
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
                    continue

                # fallback: attribute style
                u = getattr(chunk, "usage", None)
                if u is not None and not usage_emitted:
                    usage_emitted = True
                    uu = self._to_usage(u)
                    set_usage(uu)
                    yield StreamEvent(type=Evt.usage, usage=uu)
                if not getattr(chunk, "choices", None):
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                if delta is None:
                    continue
                if getattr(delta, "content", None):
                    yield StreamEvent(type=Evt.content, delta=delta.content)
                if getattr(delta, "reasoning_content", None) and req.return_thinking:
                    yield StreamEvent(type=Evt.thinking, delta=delta.reasoning_content)
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        fn = getattr(tc, "function", None)
                        tid = getattr(tc, "id", None) or last_tid or "_last"
                        if getattr(tc, "id", None):
                            last_tid = tid
                        
                        # ⭐ 首次通知 - tool_call_start
                        if tid not in notified_tools:
                            tool_name = getattr(fn, "name", None) if fn else None
                            if tool_name:
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
                        if fn is not None:
                            if getattr(fn, "name", None):
                                entry["name"] += fn.name
                            args_delta = getattr(fn, "arguments", None)
                            if args_delta:
                                # ⭐ 参数增量 - tool_call_delta
                                yield StreamEvent(
                                    type=Evt.tool_call_delta,
                                    tool_call_delta={
                                        "call_id": tid,
                                        "arguments_delta": args_delta,
                                    }
                                )
                                
                                entry["arguments"] += args_delta
                                
                                # ⭐ 进度通知 - tool_call_progress
                                current_size = len(entry["arguments"])
                                prev_size = last_progress.get(tid, 0)
                                
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
            # 流式结束时统一输出聚合后的工具调用
            for _, v in partial_tools.items():
                yield StreamEvent(type=Evt.tool_call, tool_call=normalize_tool_call(v))
            yield StreamEvent(type=Evt.completed)
        except Exception as e:
            raise TransportError(str(e)) from e
