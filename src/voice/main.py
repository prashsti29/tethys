#!/usr/bin/env python3
import asyncio
import logging
import sys
from typing import List

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioParams

from src.voice.pipeline import create_piper_tts, create_whisper_stt
from src.orchestrator import VoiceAssistantOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("voice.main")

PATIENT_NAME = "Patient"


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

    stt = create_whisper_stt()
    tts = create_piper_tts()
    orchestrator_proc = OrchestratorProcessor(patient_name=PATIENT_NAME)

    pipeline = Pipeline([
        transport.input(),          # Mic input
        stt,                        # Whisper STT → TranscriptionFrame
        orchestrator_proc,          # Orchestrator → TextFrame
        tts,                        # Piper TTS → AudioRawFrame
        transport.output(),         # Speaker output
    ])

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("\n[Voice Assistant] Session ended by user.")


if __name__ == "__main__":
    asyncio.run(main())
