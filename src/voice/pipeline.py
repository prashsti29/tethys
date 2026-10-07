import asyncio
import logging
from pathlib import Path
from typing import List, Optional

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
from src.telemetry.latency import LatencyTracer

logger = logging.getLogger(__name__)


class TranscriptToTextProcessor(FrameProcessor):
    """Collects STT transcriptions and forwards them downstream."""

    def __init__(self, latency_tracer: Optional[LatencyTracer] = None):
        super().__init__()
        self.transcriptions: List[str] = []
        self._tracer = latency_tracer

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                # Stamp: STT produced a final transcript
                if self._tracer:
                    self._tracer.record("stt_final")
                logger.info(f"[STT Transcript]: {text}")
                self.transcriptions.append(text)
                # Forward the TranscriptionFrame for downstream processors
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class OrchestratorProcessor(FrameProcessor):
    """Intercepts TranscriptionFrames, runs the medical orchestrator pipeline,
    and emits a TextFrame with the patient-facing response. Supports task cancellation on barge-in."""

    def __init__(
        self,
        patient_name: str = "Patient",
        latency_tracer: Optional[LatencyTracer] = None,
        barge_in_controller: Optional[object] = None,
    ):
        super().__init__()
        self.orchestrator = VoiceAssistantOrchestrator()
        self.patient_name = patient_name
        self.conversation_history: List[str] = []
        self._tracer = latency_tracer
        self.barge_in_controller = barge_in_controller
        self._current_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return

            # Check if this transcript constitutes an interruption
            if self.barge_in_controller and hasattr(self.barge_in_controller, "should_interrupt"):
                if self.barge_in_controller.should_interrupt(text):
                    logger.info("[Orchestrator] Interruption triggered! Canceling active LLM task.")
                    if self._current_task and not self._current_task.done():
                        self._current_task.cancel()

            logger.info(f"[Orchestrator] Processing: '{text}'")

            if self._tracer:
                self._tracer.record("llm_first_token")

            if self.barge_in_controller and hasattr(self.barge_in_controller, "is_assistant_speaking"):
                self.barge_in_controller.is_assistant_speaking = True

            loop = asyncio.get_event_loop()
            self._current_task = loop.create_task(
                loop.run_in_executor(
                    None,
                    lambda: self.orchestrator.run(
                        user_input=text,
                        patient_name=self.patient_name,
                        conversation_history=self.conversation_history if self.conversation_history else None,
                    ),
                )
            )

            try:
                result = await self._current_task
            except asyncio.CancelledError:
                logger.info("[Orchestrator] Turn execution cancelled due to barge-in.")
                if self.barge_in_controller and hasattr(self.barge_in_controller, "is_assistant_speaking"):
                    self.barge_in_controller.is_assistant_speaking = False
                return
            finally:
                self._current_task = None

            if self.barge_in_controller and hasattr(self.barge_in_controller, "is_assistant_speaking"):
                self.barge_in_controller.is_assistant_speaking = False

            # Track conversation history for multi-turn
            self.conversation_history.append(f"Patient: {text}")
            self.conversation_history.append(f"Assistant: {result.response_text}")

            logger.info(f"[Orchestrator] Response: '{result.response_text}'")
            if result.awaiting_patient_reply:
                logger.info("[Orchestrator] Awaiting patient reply for next turn")

            await self.push_frame(TextFrame(text=result.response_text))
        else:
            await self.push_frame(frame, direction)



class TTSLatencyStamper(FrameProcessor):
    """Stamps 'tts_first_byte' on the first AudioRawFrame seen in a turn."""

    def __init__(self, latency_tracer: Optional[LatencyTracer] = None):
        super().__init__()
        self._tracer = latency_tracer
        self._stamped = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame) and not self._stamped and self._tracer:
            self._tracer.record("tts_first_byte")
            self._stamped = True
        await self.push_frame(frame, direction)

    def reset(self) -> None:
        self._stamped = False


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
