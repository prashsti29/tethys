import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenException(Exception):
    """Raised when an API request is attempted while the circuit breaker is OPEN."""
    pass


class ResilientServiceAdapter:
    """
    Wraps external service calls with exponential backoff retry, circuit breaking,
    and automatic failover recovery.
    """

    def __init__(
        self,
        service_name: str,
        max_retries: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self.service_name = service_name
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    async def execute(
        self,
        func: Callable[..., Any],
        *args,
        fallback_func: Optional[Callable[..., Any]] = None,
        **kwargs,
    ) -> Any:
        """Executes a function with retry, circuit breaking, and fallback protection."""
        self._check_circuit_state()

        if self.state == CircuitState.OPEN:
            logger.warning(f"[{self.service_name}] Circuit breaker OPEN. Tripping to fallback.")
            if fallback_func:
                return await self._call_fallback(fallback_func, *args, **kwargs)
            raise CircuitBreakerOpenException(f"[{self.service_name}] Circuit open and no fallback provided.")

        delay = self.initial_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                logger.error(f"[{self.service_name}] Attempt {attempt}/{self.max_retries} failed: {e}")
                self._on_failure()
                if attempt == self.max_retries:
                    if fallback_func:
                        logger.info(f"[{self.service_name}] All retries exhausted. Executing fallback handler.")
                        return await self._call_fallback(fallback_func, *args, **kwargs)
                    raise e
                await asyncio.sleep(delay)
                delay *= self.backoff_factor

    def _check_circuit_state(self):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"[{self.service_name}] Circuit transition to HALF_OPEN (probing health)")

    def _on_success(self):
        if self.state != CircuitState.CLOSED:
            logger.info(f"[{self.service_name}] Circuit recovered! Resetting to CLOSED.")
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"[{self.service_name}] Failure threshold reached ({self.failure_count}). Tripping circuit OPEN!")

    async def _call_fallback(self, fallback_func: Callable[..., Any], *args, **kwargs) -> Any:
        if asyncio.iscoroutinefunction(fallback_func):
            return await fallback_func(*args, **kwargs)
        return fallback_func(*args, **kwargs)
