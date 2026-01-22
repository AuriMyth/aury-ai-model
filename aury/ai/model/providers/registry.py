from __future__ import annotations
from typing import Type
from .openai import OpenAIAdapter
from .openrouter import OpenRouterAdapter
from .doubao import DoubaoArkAdapter

_DEF_BASE = {
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
}

_ADAPTERS = {
    "openai": OpenAIAdapter,
    "openrouter": OpenRouterAdapter,
    "doubao": DoubaoArkAdapter,
}

def make_adapter(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    headers: dict | None = None,
    transport: str | None = None,
    timeout: float | None = None,
):
    p = provider.lower()
    if p not in _ADAPTERS:
        raise ValueError(f"Unsupported provider: {provider}")
    adapter_cls: Type = _ADAPTERS[p]
    base = base_url or _DEF_BASE.get(p)
    # Try with all parameters, fall back if adapter doesn't support some
    try:
        return adapter_cls(
            model=model,
            base_url=base,
            api_key=api_key,
            headers=headers or {},
            transport=transport,
            timeout=timeout,
        )
    except TypeError:
        # Fallback: try without transport and timeout
        try:
            return adapter_cls(
                model=model,
                base_url=base,
                api_key=api_key,
                headers=headers or {},
                timeout=timeout,
            )
        except TypeError:
            return adapter_cls(
                model=model,
                base_url=base,
                api_key=api_key,
                headers=headers or {},
            )
