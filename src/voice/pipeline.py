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
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.task import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService

from src.config import settings
from src.orchestrator import VoiceAssistantOrchestrator

logger = logging.getLogger(__name__)


class TranscriptToTextProcessor(FrameProcessor):
    """Collects STT transcriptions and forwards them downstream."""

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
                # Forward the TranscriptionFrame for downstream processors
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class OrchestratorProcessor(FrameProcessor):
    """Intercepts TranscriptionFrames, runs the medical orchestrator pipeline,
    and emits a TextFrame with the patient-facing response."""

    def __init__(self, patient_name: str = "Patient"):
        super().__init__()
        self.orchestrator = VoiceAssistantOrchestrator()
        self.patient_name = patient_name
        self.conversation_history: List[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return

            logger.info(f"[Orchestrator] Processing: '{text}'")

            # Run orchestrator in a thread to avoid blocking the async pipeline
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.orchestrator.run(
                    user_input=text,
                    patient_name=self.patient_name,
                    conversation_history=self.conversation_history if self.conversation_history else None,
                ),
            )

            # Track conversation history for multi-turn
            self.conversation_history.append(f"Patient: {text}")
            self.conversation_history.append(f"Assistant: {result.response_text}")

            logger.info(f"[Orchestrator] Response: '{result.response_text}'")
            if result.awaiting_patient_reply:
                logger.info("[Orchestrator] Awaiting patient reply for next turn")

            # Emit the response as a TextFrame for TTS
            await self.push_frame(TextFrame(text=result.response_text))
        else:
            await self.push_frame(frame, direction)


class AudioSaverProcessor(FrameProcessor):

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
