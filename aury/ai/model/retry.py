from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Tuple, Type

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .types import Message, StreamEvent


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 0.5  # seconds
    max_delay: float = 5.0
    backoff_factor: float = 2.0
    retry_on: Tuple[Type[BaseException], ...] | None = None
    predicate: Callable[[BaseException], bool] | None = None


class RetryView:
    """A tenacity-based retrying view for ModelClient.

    Usage:
        client.with_retry(max_attempts=3).ainvoke(...)
        async for ev in client.with_retry().astream(...): ...
    """

    def __init__(self, client: "ModelClient", cfg: RetryConfig):
        self.client = client
        self.cfg = cfg

    def _retry_condition(self):
        cond = None
        if self.cfg.retry_on:
            cond = retry_if_exception_type(self.cfg.retry_on)
        if self.cfg.predicate:
            pred_cond = retry_if_exception(self.cfg.predicate)
            cond = pred_cond if cond is None else (cond | pred_cond)
        # Fallback: if neither provided, never retry (caller should have provided retry_on)
        return cond or retry_if_exception(lambda e: False)

    async def ainvoke(self, messages: list[Message], **kw) -> Message:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.cfg.max_attempts),
            wait=wait_exponential(
                multiplier=self.cfg.base_delay,
                min=self.cfg.base_delay,
                max=self.cfg.max_delay,
                exp_base=self.cfg.backoff_factor,
            ),
            retry=self._retry_condition(),
            reraise=True,
        ):
            with attempt:
                return await self.client.ainvoke(messages, **kw)
        # Unreachable due to reraise=True

    async def astream(self, messages: list[Message], **kw) -> AsyncIterator[StreamEvent]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.cfg.max_attempts),
            wait=wait_exponential(
                multiplier=self.cfg.base_delay,
                min=self.cfg.base_delay,
                max=self.cfg.max_delay,
                exp_base=self.cfg.backoff_factor,
            ),
            retry=self._retry_condition(),
            reraise=True,
        ):
            with attempt:
                async for ev in self.client.astream(messages, **kw):
                    yield ev
                return