import asyncio
import logging
from typing import Optional, Callable
import websockets
from websockets.server import WebSocketServerProtocol

from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


class WebSocketAudioOutputProcessor(FrameProcessor):
    """
    Pipeline processor that intercepts outgoing AudioRawFrames and forwards them to a connected WebSocket client.
    """

    def __init__(self, websocket: WebSocketServerProtocol):
        super().__init__()
        self.websocket = websocket

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            try:
                await self.websocket.send(frame.audio)
            except Exception as e:
                logger.error(f"Failed to send binary audio over WebSocket: {e}")
        
        await self.push_frame(frame, direction)


class AudioWebSocketServer:
    """
    Real-time binary audio WebSocket server for streaming raw audio (16kHz 16-bit PCM) back and forth.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._server: Optional[websockets.WebSocketServer] = None

    async def handle_connection(
        self,
        websocket: WebSocketServerProtocol,
        on_audio_frame: Optional[Callable[[bytes], None]] = None,
    ):
        logger.info(f"New WebSocket audio client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if on_audio_frame:
                        if asyncio.iscoroutinefunction(on_audio_frame):
                            await on_audio_frame(message)
                        else:
                            on_audio_frame(message)
                else:
                    logger.debug(f"Received non-binary text frame on audio socket: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling WebSocket audio client: {e}")

    async def start(self, handler_callback: Callable):
        self._server = await websockets.serve(
            lambda ws: self.handle_connection(ws, on_audio_frame=handler_callback),
            self.host,
            self.port,
        )
        logger.info(f"Audio WebSocket Server running on ws://{self.host}:{self.port}")

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Audio WebSocket Server stopped.")
