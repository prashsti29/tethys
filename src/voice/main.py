#!/usr/bin/env python3
"""
Live voice pipeline entry-point with per-turn latency instrumentation.

Timestamps every stage  (VAD end → STT final → LLM first token →
TTS first byte → playback start)  and reports P50/P95 against budget.
"""

import asyncio
import logging
import sys
from typing import List, Optional

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.workers.runner import WorkerRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioParams

from src.config import settings
from src.orchestrator import VoiceAssistantOrchestrator
from src.telemetry.latency import LatencyAggregator, LatencyTracer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("voice.main")

PATIENT_NAME = "Patient"

# ── Shared latency state ─────────────────────────────────────────
_aggregator = LatencyAggregator()
_current_tracer: Optional[LatencyTracer] = None
_turn_counter: int = 0


def _new_tracer() -> LatencyTracer:
    """Create and register a fresh tracer for the next turn."""
    global _current_tracer, _turn_counter
    _turn_counter += 1
    _current_tracer = LatencyTracer(turn_id=_turn_counter)
    return _current_tracer


def _finish_tracer() -> None:
    """Log the current tracer and ingest into the aggregator."""
    global _current_tracer
    if _current_tracer is None:
        return
    if settings.LATENCY_LOGGING_ENABLED:
        _current_tracer.log_report()
    _aggregator.ingest(_current_tracer)
    if _turn_counter % settings.LATENCY_AGGREGATE_INTERVAL == 0:
        _aggregator.log_summary()
    _current_tracer = None


# ── VAD Stamper ──────────────────────────────────────────────────

class VADLatencyStamper(FrameProcessor):
    """
    Sits right after the transport input.
    On VADUserStoppedSpeakingFrame, creates a new LatencyTracer and
    stamps ``vad_end``.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Pipecat fires this frame when the VAD detects end-of-speech
        frame_name = type(frame).__name__
        if frame_name == "VADUserStoppedSpeakingFrame":
            tracer = _new_tracer()
            tracer.record("vad_end")
            logger.debug(f"[VADLatencyStamper] vad_end stamped (turn #{tracer.turn_id})")

        await self.push_frame(frame, direction)


# ── STT Stamper ──────────────────────────────────────────────────

class STTLatencyStamper(FrameProcessor):
    """Stamps ``stt_final`` on the first TranscriptionFrame after vad_end."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and _current_tracer:
            _current_tracer.record("stt_final")

        await self.push_frame(frame, direction)


# ── TTS Stamper ──────────────────────────────────────────────────

class TTSLatencyStamper(FrameProcessor):
    """Stamps ``tts_first_byte`` on the first AudioRawFrame per turn."""

    def __init__(self):
        super().__init__()
        self._stamped = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame) and not self._stamped and _current_tracer:
            _current_tracer.record("tts_first_byte")
            self._stamped = True

        await self.push_frame(frame, direction)

    def reset(self):
        self._stamped = False


# ── Playback Stamper ─────────────────────────────────────────────

class PlaybackLatencyStamper(FrameProcessor):
    """Stamps ``playback_start`` on the first AudioRawFrame reaching the
    output transport and then finalises the tracer for this turn."""

    def __init__(self):
        super().__init__()
        self._stamped = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame) and not self._stamped:
            if _current_tracer:
                _current_tracer.record("playback_start")
            self._stamped = True
            # Finalize the tracer — all stages have been stamped
            _finish_tracer()

        await self.push_frame(frame, direction)

    def reset(self):
        self._stamped = False


# ── Orchestrator Processor (with LLM stamp) ─────────────────────

class OrchestratorProcessor(FrameProcessor):
    """
    Intercepts TranscriptionFrame from Whisper STT,
    runs text through VoiceAssistantOrchestrator,
    and pushes the response back as a TextFrame for Piper TTS to synthesize.
    """

    def __init__(self, patient_name: str = PATIENT_NAME):
        super().__init__()
        self.patient_name = patient_name
        self.orchestrator = VoiceAssistantOrchestrator()
        self.conversation_history: List[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            if not user_text:
                return

            logger.info(f"\n[USER] {user_text}")
            self.conversation_history.append(f"Patient: {user_text}")

            # Stamp: LLM / orchestrator work begins
            if _current_tracer:
                _current_tracer.record("llm_first_token")

            try:
                result = self.orchestrator.run(
                    user_input=user_text,
                    patient_name=self.patient_name,
                    conversation_history=self.conversation_history[-6:],   # last 3 turns
                    auto_book=True,
                )
            except Exception as e:
                logger.error(f"[OrchestratorProcessor] Error: {e}")
                result = None

            response = (
                result.response_text if result
                else "I'm sorry, I had trouble understanding you. Could you please repeat that?"
            )

            logger.info(f"[ASSISTANT] {response}")
            self.conversation_history.append(f"Assistant: {response}")

            # Push response text to Piper TTS
            await self.push_frame(TextFrame(text=response))

            # If conversation is over (no follow-up), optionally push EndFrame
            if result and not result.awaiting_patient_reply and result.booking and result.booking.success:
                logger.info("[OrchestratorProcessor] Booking complete. You may close the app or continue speaking.")
        else:
            await self.push_frame(frame, direction)


# ── Main ─────────────────────────────────────────────────────────

async def main():
    logger.info("=== Voice Medical Assistant Starting ===")
    logger.info("Speak into your microphone. Press Ctrl+C to stop.\n")

    transport = LocalAudioTransport(
        LocalAudioParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # Pipeline stages
    vad_stamper = VADLatencyStamper()
    stt = __import__("src.voice.pipeline", fromlist=["create_whisper_stt"]).create_whisper_stt()
    stt_stamper = STTLatencyStamper()
    orchestrator_proc = OrchestratorProcessor(patient_name=PATIENT_NAME)
    tts = __import__("src.voice.pipeline", fromlist=["create_piper_tts"]).create_piper_tts()
    tts_stamper = TTSLatencyStamper()
    playback_stamper = PlaybackLatencyStamper()

    pipeline = Pipeline([
        transport.input(),          # Mic input
        vad_stamper,                # ── stamp vad_end
        stt,                        # Whisper STT → TranscriptionFrame
        stt_stamper,                # ── stamp stt_final
        orchestrator_proc,          # Orchestrator (stamps llm_first_token) → TextFrame
        tts,                        # Piper TTS → AudioRawFrame
        tts_stamper,                # ── stamp tts_first_byte
        playback_stamper,           # ── stamp playback_start + finalize tracer
        transport.output(),         # Speaker output
    ])

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    runner = WorkerRunner()

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("\n[Voice Assistant] Session ended by user.")
        # Print final aggregate latency summary
        _aggregator.log_summary()


if __name__ == "__main__":
    asyncio.run(main())
