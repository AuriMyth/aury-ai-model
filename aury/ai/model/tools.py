from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, Field, ConfigDict
from typing import Any
import re
from .types import ToolCall

class ToolKind(StrEnum):
    function = "function"
    mcp = "mcp"
    builtin = "builtin"

class FunctionToolSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool = True

class MCPToolSpec(BaseModel):
    server_id: str
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    result_schema: dict[str, Any] | None = None

class BuiltinToolSpec(BaseModel):
    type: str
    config: dict[str, Any] = Field(default_factory=dict)

class ToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: ToolKind
    function: FunctionToolSpec | None = None
    mcp: MCPToolSpec | None = None
    builtin: BuiltinToolSpec | None = None

# ---- mapping helpers ----

def to_openai_tools(tools: list[ToolSpec], *, supports_mcp_native: bool=False) -> list[dict]:
    out: list[dict] = []
    for t in tools:
        if t.kind == ToolKind.function and t.function:
            out.append({"type":"function","function":{
                "name": t.function.name,
                "description": t.function.description or "",
                "parameters": t.function.parameters,
            }})
        elif t.kind == ToolKind.mcp and t.mcp:
            if supports_mcp_native:
                out.append({"type":"mcp","server": t.mcp.server_id,
                            "name": t.mcp.name, "parameters": t.mcp.input_schema})
            else:
                enc = f"mcp::{t.mcp.server_id}::{t.mcp.name}"
                desc = (t.mcp.description or "") + f" [MCP server={t.mcp.server_id}]"
                out.append({"type":"function","function":{
                    "name": enc, "description": desc, "parameters": t.mcp.input_schema
                }})
        elif t.kind == ToolKind.builtin and t.builtin:
            item = {"type": t.builtin.type}
            item.update(t.builtin.config)
            out.append(item)
    return out

_mcp_pat = re.compile(r"^mcp::(.+?)::(.+)$")

def decode_maybe_mcp(name: str) -> tuple[str|None, str]:
    m = _mcp_pat.match(name)
    return (m.group(1), m.group(2)) if m else (None, name)


def normalize_tool_call(raw: dict) -> ToolCall:
    # raw: {id, name, arguments}
    server, tool = decode_maybe_mcp(raw.get("name",""))
    return ToolCall(
        id=raw.get("id") or "",
        name=tool if server else raw.get("name",""),
        arguments_json=raw.get("arguments","{}"),
        mcp_server_id=server,
    )
