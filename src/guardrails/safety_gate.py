"""
Safety gate — runs guardrails on a token hold-back buffer before TTS,
so unsafe content is never spoken aloud.

An emergency-symptom detector can hard-interrupt the agent mid-stream.
"""

import logging
import re
from collections import deque
from typing import Deque, List, Optional

from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from src.config import settings
from src.guardrails.red_flags import CLINICAL_RULES

logger = logging.getLogger(__name__)

# Patterns that should trigger an immediate hard-interrupt of the agent
EMERGENCY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for rule in CLINICAL_RULES
    for p in rule["patterns"]
]

# Phrases the agent should never speak (unsafe medical advice, etc.)
UNSAFE_AGENT_PATTERNS = [
    re.compile(r"take\s+\d+\s*(?:mg|milligrams?)\s+of", re.IGNORECASE),
    re.compile(r"prescribe\s+(?:you|yourself)", re.IGNORECASE),
    re.compile(r"you\s+(?:should|can)\s+stop\s+taking\s+(?:your\s+)?(?:medication|medicine)", re.IGNORECASE),
    re.compile(r"diagnos(?:e|is|ing)\s+you\s+with", re.IGNORECASE),
    re.compile(r"you\s+(?:have|definitely\s+have)\s+(?:cancer|diabetes|HIV|AIDS)", re.IGNORECASE),
]

SAFE_FALLBACK = (
    "I want to make sure I give you the most helpful information. "
    "Let me connect you with a healthcare professional who can better assist you."
)

EMERGENCY_INTERRUPT = (
    "I'm detecting potential emergency symptoms in what you've described. "
    "Please call 911 or go to your nearest emergency room immediately."
)


class SafetyGateProcessor(FrameProcessor):
    """
    Sits between the LLM/orchestrator output and TTS.

    Holds back a sliding window of tokens (TextFrames). Before releasing
    tokens to TTS, runs them through safety checks:
      1. Unsafe agent output patterns → replaced with safe fallback.
      2. Emergency symptom detector on patient context → hard interrupts.
    """

    def __init__(self, holdback_size: Optional[int] = None) -> None:
        super().__init__()
        self._holdback_size = holdback_size or settings.SAFETY_GATE_HOLDBACK_TOKENS
        self._buffer: Deque[str] = deque()
        self._enabled = settings.SAFETY_GATE_ENABLED
        self._emergency_triggered = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not self._enabled:
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, TextFrame):
            # Non-text frames pass through (audio, control frames, etc.)
            await self.push_frame(frame, direction)
            return

        text = frame.text
        if not text:
            return

        # Add incoming text tokens to the buffer
        words = text.split()
        self._buffer.extend(words)

        # Once buffer exceeds hold-back size, release safe tokens
        while len(self._buffer) > self._holdback_size:
            released_word = self._buffer.popleft()
            await self.push_frame(TextFrame(text=released_word + " "), direction)

        # Periodically check the buffer content for unsafe patterns
        buffer_text = " ".join(self._buffer)
        if self._check_unsafe_agent_output(buffer_text):
            logger.warning(
                f"[SafetyGate] Unsafe agent output detected, replacing buffer: '{buffer_text[:80]}...'"
            )
            self._buffer.clear()
            await self.push_frame(TextFrame(text=SAFE_FALLBACK), direction)
            return

    def check_patient_emergency(self, patient_text: str) -> bool:
        """
        Check if patient speech contains emergency symptoms.
        Called externally by the barge-in handler on patient transcriptions.

        Returns True if emergency detected (caller should hard-interrupt).
        """
        for pattern in EMERGENCY_PATTERNS:
            if pattern.search(patient_text):
                logger.warning(
                    f"[SafetyGate] Emergency symptom detected in patient speech: "
                    f"'{patient_text[:80]}...'"
                )
                self._emergency_triggered = True
                return True
        return False

    def _check_unsafe_agent_output(self, text: str) -> bool:
        """Check if buffered agent output contains unsafe medical content."""
        for pattern in UNSAFE_AGENT_PATTERNS:
            if pattern.search(text):
                return True
        return False

    async def flush_safe(self) -> None:
        """Flush remaining buffer content to TTS (call at end of turn)."""
        if not self._buffer:
            return

        buffer_text = " ".join(self._buffer)
        if self._check_unsafe_agent_output(buffer_text):
            logger.warning("[SafetyGate] Unsafe content in final flush, replacing with fallback.")
            self._buffer.clear()
            await self.push_frame(TextFrame(text=SAFE_FALLBACK))
            return

        # Safe — release everything
        remaining = " ".join(self._buffer)
        self._buffer.clear()
        if remaining.strip():
            await self.push_frame(TextFrame(text=remaining))

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._buffer.clear()
        self._emergency_triggered = False

    @property
    def emergency_triggered(self) -> bool:
        return self._emergency_triggered
