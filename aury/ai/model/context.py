from __future__ import annotations
from contextvars import ContextVar, Token
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
from inspect import iscoroutinefunction
from pydantic import BaseModel, ConfigDict

class RequestContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    trace_id: str | None = None
    request_id: str | None = None
    user_id: str | None = None
    provider: str | None = None
    model: str | None = None
    extra_headers: dict = {}

_ctx: ContextVar[RequestContext] = ContextVar("ctx", default=RequestContext())

def get_ctx() -> RequestContext:
    return _ctx.get()

# --- ergonomic helpers ---

def push_ctx(**updates) -> Token:
    """Push updates to context; returns a Token for later pop.
    Usage:
        tok = push_ctx(trace_id="t1"); ...; pop_ctx(tok)
    """
    return _ctx.set(get_ctx().model_copy(update=updates))

def pop_ctx(token: Token) -> None:
    _ctx.reset(token)

@contextmanager
def set_ctx(**updates):
    old = _ctx.set(get_ctx().model_copy(update=updates))
    try:
        yield
    finally:
        try:
            _ctx.reset(old)
        except (LookupError, ValueError):
            # Context may have been changed by another task or in different async context
            pass

@asynccontextmanager
async def aset_ctx(**updates):
    old = _ctx.set(get_ctx().model_copy(update=updates))
    try:
        yield
    finally:
        try:
            _ctx.reset(old)
        except (LookupError, ValueError):
            # Context may have been changed by another task or in different async context
            pass

# Library-unique context manager & decorator
class model_ctx:
    """Model layer context helper - can be used as context manager or decorator.
    Usage:
      - with model_ctx(trace_id="..."):
      - @model_ctx(trace_id="...") def f(...): ...
      - @model_ctx(trace_id="...") async def g(...): ...
    """
    def __init__(self, **updates):
        self._updates = updates
        self._token: Token | None = None
    def __enter__(self):
        self._token = push_ctx(**self._updates)
        return self
    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            pop_ctx(self._token)
        return False
    async def __aenter__(self):
        self.__enter__()
        return self
    async def __aexit__(self, exc_type, exc, tb):
        self.__exit__(exc_type, exc, tb)
    def __call__(self, fn):
        if iscoroutinefunction(fn):
            @wraps(fn)
            async def aw(*a, **kw):
                tok = push_ctx(**self._updates)
                try:
                    return await fn(*a, **kw)
                finally:
                    pop_ctx(tok)
            return aw
        else:
            @wraps(fn)
            def w(*a, **kw):
                tok = push_ctx(**self._updates)
                try:
                    return fn(*a, **kw)
                finally:
                    pop_ctx(tok)
            return w

