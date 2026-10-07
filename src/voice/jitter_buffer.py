import asyncio
import logging
from typing import Optional

from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


class JitterBufferProcessor(FrameProcessor):
    """
    Adaptive Jitter Buffer processor to smooth out uneven raw audio packet arrivals
    in real-time audio transport streams.
    """

    def __init__(
        self,
        target_delay_ms: int = 60,
        sample_rate: int = 16000,
        num_channels: int = 1,
        bytes_per_sample: int = 2,
    ):
        super().__init__()
        self.target_delay_ms = target_delay_ms
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.bytes_per_sample = bytes_per_sample

        # Calculate target buffer threshold in bytes
        self.target_buffer_size = int(
            (sample_rate * num_channels * bytes_per_sample * target_delay_ms) / 1000
        )
        self._buffer = bytearray()
        self._is_buffering = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            self._buffer.extend(frame.audio)
            
            # Initial buffering phase to smooth initial packet arrival
            if self._is_buffering:
                if len(self._buffer) >= self.target_buffer_size:
                    self._is_buffering = False
                else:
                    return

            # Emit standard frame chunk sizes (e.g. 20ms audio frames = 640 bytes at 16kHz mono 16-bit)
            chunk_size = int((self.sample_rate * self.num_channels * self.bytes_per_sample * 20) / 1000)
            while len(self._buffer) >= chunk_size:
                chunk = bytes(self._buffer[:chunk_size])
                self._buffer = self._buffer[chunk_size:]
                out_frame = AudioRawFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=self.num_channels,
                )
                await self.push_frame(out_frame, direction)
        else:
            # Non-audio frames pass through immediately
            await self.push_frame(frame, direction)

    def reset(self):
        """Flushes the jitter buffer and resets buffering state."""
        self._buffer.clear()
        self._is_buffering = True
