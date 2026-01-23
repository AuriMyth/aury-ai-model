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

def _strip_extension_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip non-standard JSON Schema extension fields (x-* prefixed).
    
    Some providers (e.g. Gemini) reject schemas containing custom extensions
    like 'x-file', 'x-custom', etc.
    """
    if not isinstance(schema, dict):
        return schema
    
    result = {}
    for k, v in schema.items():
        # Skip x-* extension fields
        if isinstance(k, str) and k.startswith("x-"):
            continue
        # Recursively process nested dicts
        if isinstance(v, dict):
            result[k] = _strip_extension_fields(v)
        elif isinstance(v, list):
            result[k] = [_strip_extension_fields(item) if isinstance(item, dict) else item for item in v]
        else:
            result[k] = v
    return result


def _normalize_schema(schema: dict[str, Any], *, strip_extensions: bool = False) -> dict[str, Any]:
    """Normalize JSON Schema to be compatible with JSON Schema 2020-12.
    
    Ensures:
    - Every schema has a 'type' field
    - object type has 'properties' field
    - object type has 'required' field
    - nested schemas are also normalized
    
    Args:
        schema: The schema to normalize
        strip_extensions: If True, remove x-* extension fields (for Gemini compatibility)
    """
    if not isinstance(schema, dict):
        return schema
    
    # First strip extensions if needed
    if strip_extensions:
        schema = _strip_extension_fields(schema)
    
    result = dict(schema)
    
    # Ensure type field exists and is valid (JSON Schema 2020-12 requirement)
    # Valid types: string, number, integer, boolean, object, array, null
    valid_types = {"string", "number", "integer", "boolean", "object", "array", "null"}
    
    if "type" not in result:
        # Infer type from other fields or default to string
        if "properties" in result or "additionalProperties" in result:
            result["type"] = "object"
        elif "items" in result:
            result["type"] = "array"
        elif "enum" in result:
            # enum can be any type, keep as-is (type is optional for enum)
            pass
        else:
            result["type"] = "string"  # Safe default
    elif result["type"] not in valid_types:
        # Convert invalid types (e.g. "file") to string
        result["type"] = "string"
    
    # If type is object, ensure properties and required exist
    if result.get("type") == "object":
        if "properties" not in result:
            result["properties"] = {}
        if "required" not in result:
            result["required"] = []
        # Normalize nested property schemas
        if isinstance(result.get("properties"), dict):
            result["properties"] = {
                k: _normalize_schema(v) for k, v in result["properties"].items()
            }
    
    # Normalize array items
    if result.get("type") == "array" and "items" in result:
        result["items"] = _normalize_schema(result["items"], strip_extensions=strip_extensions)
    
    # Normalize additionalProperties if it's a schema
    if isinstance(result.get("additionalProperties"), dict):
        result["additionalProperties"] = _normalize_schema(result["additionalProperties"], strip_extensions=strip_extensions)
    
    return result


def _is_gemini_model(model: str | None) -> bool:
    """Check if the model is a Gemini model that requires schema sanitization."""
    if not model:
        return False
    model_lower = model.lower()
    return "gemini" in model_lower or "google/" in model_lower


def to_openai_tools(tools: list[ToolSpec], *, supports_mcp_native: bool = False, model: str | None = None) -> list[dict]:
    """Convert ToolSpec list to OpenAI-compatible tools format.
    
    Args:
        tools: List of ToolSpec to convert
        supports_mcp_native: Whether the provider supports native MCP tools
        model: Model name (used to determine if schema sanitization is needed, e.g. for Gemini)
    """
    # Gemini models reject x-* extension fields in schemas
    strip_extensions = _is_gemini_model(model)
    
    out: list[dict] = []
    for t in tools:
        if t.kind == ToolKind.function and t.function:
            out.append({"type":"function","function":{
                "name": t.function.name,
                "description": t.function.description or "",
                "parameters": _normalize_schema(t.function.parameters, strip_extensions=strip_extensions),
            }})
        elif t.kind == ToolKind.mcp and t.mcp:
            if supports_mcp_native:
                out.append({"type":"mcp","server": t.mcp.server_id,
                            "name": t.mcp.name, "parameters": _normalize_schema(t.mcp.input_schema, strip_extensions=strip_extensions)})
            else:
                enc = f"mcp::{t.mcp.server_id}::{t.mcp.name}"
                desc = (t.mcp.description or "") + f" [MCP server={t.mcp.server_id}]"
                out.append({"type":"function","function":{
                    "name": enc, "description": desc, "parameters": _normalize_schema(t.mcp.input_schema, strip_extensions=strip_extensions)
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
