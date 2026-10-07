import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Set

logger = logging.getLogger(__name__)


class SSEControlGateway:
    """
    Server-Sent Events (SSE) control gateway for real-time activity signaling,
    telemetry metrics, and UI state synchronization.
    """

    def __init__(self):
        self._subscribers: Set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        """Subscribes a client event listener queue."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        logger.info(f"[SSEGateway] Client subscribed. Total active listeners: {len(self._subscribers)}")
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribes a client event listener queue."""
        self._subscribers.discard(queue)
        logger.info(f"[SSEGateway] Client unsubscribed. Remaining listeners: {len(self._subscribers)}")

    async def broadcast_event(self, event_type: str, data: Dict) -> None:
        """Broadcasts a JSON control event to all subscribed client UI listeners."""
        payload = json.dumps({"event": event_type, "data": data})
        for queue in list(self._subscribers):
            try:
                await queue.put(payload)
            except Exception as e:
                logger.error(f"[SSEGateway] Error broadcasting event to subscriber: {e}")

    async def event_generator(self, queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """Async generator emitting formatted SSE events."""
        try:
            while True:
                payload = await queue.get()
                yield f"data: {payload}\n\n"
        except asyncio.CancelledError:
            self.unsubscribe(queue)
