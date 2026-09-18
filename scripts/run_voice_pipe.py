#!/usr/bin/env python3
import asyncio
import logging
import sys
import wave
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from src.voice.pipeline import (
    AudioSaverProcessor,
    TranscriptToTextProcessor,
    create_piper_tts,
    create_whisper_stt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("voice_pipe_test")


def resample_audio(audio_bytes: bytes, orig_sr: int, target_sr: int = 16000) -> bytes:
    if orig_sr == target_sr:
        return audio_bytes

    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    num_output_samples = int(len(audio_data) * target_sr / orig_sr)
    resampled_data = np.interp(
        np.linspace(0, len(audio_data), num_output_samples, endpoint=False),
        np.arange(len(audio_data)),
        audio_data,
    ).astype(np.int16)
    return resampled_data.tobytes()


async def main():
    test_wav_path = Path("test.wav")
    output_wav_path = Path("output_response.wav")

    if not test_wav_path.exists():
        logger.error(f"Test audio file not found: {test_wav_path}")
        return

    logger.info("--- Initializing Voice Pipeline Components ---")
    stt = create_whisper_stt()
    transcript_proc = TranscriptToTextProcessor()
    tts = create_piper_tts()
    audio_saver = AudioSaverProcessor()

    pipeline = Pipeline([stt, transcript_proc, tts, audio_saver])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False))
    runner = PipelineRunner()

    logger.info("--- Reading input audio file test.wav ---")
    with wave.open(str(test_wav_path), "rb") as wf:
        orig_sr = wf.getframerate()
        raw_bytes = wf.readframes(wf.getnframes())

    target_sr = 16000
    resampled_bytes = resample_audio(raw_bytes, orig_sr, target_sr)
    logger.info(
        f"Audio resampled from {orig_sr}Hz to {target_sr}Hz (total {len(resampled_bytes)} bytes)"
    )

    async def send_audio_frames():
        # Start frame
        await task.queue_frame(StartFrame())
        await asyncio.sleep(0.05)

        # Notify STT speech started
        await task.queue_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(0.05)

        chunk_duration_sec = 0.1  # 100ms
        bytes_per_sample = 2
        chunk_bytes_len = int(target_sr * chunk_duration_sec * bytes_per_sample)

        offset = 0
        while offset < len(resampled_bytes):
            chunk = resampled_bytes[offset : offset + chunk_bytes_len]
            if not chunk:
                break
            frame = InputAudioRawFrame(
                audio=chunk,
                sample_rate=target_sr,
                num_channels=1,
            )
            await task.queue_frame(frame)
            offset += chunk_bytes_len
            await asyncio.sleep(0.02)

        # Notify STT speech stopped
        await task.queue_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(1.0)

        # End pipeline
        await task.queue_frame(EndFrame())

    logger.info("--- Starting Pipeline Execution ---")
    pipeline_runner_task = asyncio.create_task(runner.run(task))
    frame_sender_task = asyncio.create_task(send_audio_frames())

    await asyncio.gather(pipeline_runner_task, frame_sender_task)

    logger.info("--- Pipeline Completed ---")
    logger.info(f"Transcribed Text Segments: {transcript_proc.transcriptions}")

    if audio_saver.audio_bytes:
        logger.info(f"Saving synthesized TTS audio to {output_wav_path}...")
        with wave.open(str(output_wav_path), "wb") as out_wf:
            out_wf.setnchannels(audio_saver.num_channels)
            out_wf.setsampwidth(2)  # 16-bit PCM
            out_wf.setframerate(audio_saver.sample_rate)
            out_wf.writeframes(audio_saver.audio_bytes)

        logger.info(
            f"Successfully saved {len(audio_saver.audio_bytes)} bytes of TTS audio to {output_wav_path}"
        )
    else:
        logger.warning("No audio bytes produced by TTS.")


if __name__ == "__main__":
    asyncio.run(main())
