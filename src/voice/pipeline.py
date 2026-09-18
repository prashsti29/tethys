import asyncio
import logging
from pathlib import Path
from typing import List

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService

from src.config import settings

logger = logging.getLogger(__name__)


class TranscriptToTextProcessor(FrameProcessor):
    """
    Converts transcription frames into text frames to feed downstream TTS.
    Captures full text transcript for verification & logging.
    """

    def __init__(self):
        super().__init__()
        self.transcriptions: List[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                logger.info(f"[STT Transcript]: {text}")
                self.transcriptions.append(text)
                # Forward as TextFrame so TTS synthesizes response speech
                await self.push_frame(TextFrame(text=f"Echoing back: {text}"))
        else:
            await self.push_frame(frame, direction)


class AudioSaverProcessor(FrameProcessor):
    """
    Collects generated synthesized audio frames from Piper TTS into a list.
    """

    def __init__(self):
        super().__init__()
        self.audio_bytes = bytearray()
        self.sample_rate = 16000
        self.num_channels = 1

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            self.audio_bytes.extend(frame.audio)
            self.sample_rate = frame.sample_rate
            self.num_channels = frame.num_channels
        await self.push_frame(frame, direction)


def create_whisper_stt() -> WhisperSTTService:
    return WhisperSTTService(
        model=settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
    )


def create_piper_tts() -> PiperTTSService:
    model_dir = Path(settings.PIPER_MODEL_PATH).parent
    return PiperTTSService(
        voice_id="en_US-lessac-medium",
        download_dir=model_dir,
    )
