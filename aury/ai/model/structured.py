from __future__ import annotations
import json
import re
import json_repair
from typing import Protocol, TypeVar, Type, Generic
from pydantic import BaseModel, TypeAdapter
from .types import Message, Text

T = TypeVar("T", bound=BaseModel)

class StructuredStep(Protocol):
    def can_apply(self, **kw) -> bool: ...
    async def apply(self, text: str, schema: Type[T]) -> T | None: ...

class StrictSchemaStep:
    def can_apply(self, **kw) -> bool:
        # 由调用者通过 expect_strict 提示此步可用
        return bool(kw.get("expect_strict"))
    async def apply(self, text: str, schema: Type[T]) -> T | None:
        return TypeAdapter(schema).validate_json(text)

class RepairExtractStep:
    def can_apply(self, **kw) -> bool: return True
    async def apply(self, text: str, schema: Type[T]) -> T | None:
        # 1) 直接解析
        try:
            return TypeAdapter(schema).validate_json(text)
        except Exception:
            pass
        # 2) repair 再解析
        try:
            fixed = json_repair.repair_json(text)
            return TypeAdapter(schema).validate_python(json.loads(fixed))
        except Exception:
            pass
        # 3) 提取代码块/最大花括号
        m = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```|({[\s\S]*})", text)
        if m:
            return TypeAdapter(schema).validate_python(json.loads(m.group(1) or m.group(2)))
        return None

class StructuredView(Generic[T]):
    def __init__(self, client: "ModelClient", schema: Type[T], steps: list[StructuredStep] | None=None):
        self.client = client
        self.schema = schema
        self.steps = steps or [StrictSchemaStep(), RepairExtractStep()]

    async def ainvoke(self, messages: list[Message], **kw) -> T:
        m = await self.client.ainvoke(messages, **kw)
        text = "".join(p.text for p in m.parts if isinstance(p, Text))
        last: Exception | None = None
        for step in self.steps:
            if not step.can_apply(**kw):
                continue
            try:
                v = await step.apply(text, self.schema)
                if v is not None:
                    return v
            except Exception as e:
                last = e
                continue
        raise ValueError(f"structured parse failed: {last or 'no strategy matched'}")
