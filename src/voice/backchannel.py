"""
Backchannel detector — distinguishes short acknowledgments ("uh-huh", "mm-hmm")
from real interruptions so the barge-in system doesn't cancel playback on them.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)

# Keywords that indicate a backchannel rather than a real turn
BACKCHANNEL_KEYWORDS = frozenset({
    "uh-huh", "uh huh", "uhuh",
    "mm-hmm", "mm hmm", "mmhmm", "mhm",
    "okay", "ok",
    "yeah", "yep", "yes",
    "right",
    "sure",
    "got it",
    "i see",
    "go on",
    "continue",
})


@dataclass
class BackchannelResult:
    is_backchannel: bool
    text: str
    duration_ms: Optional[float] = None
    reason: str = ""


class BackchannelDetector:
    """
    Classifies a VAD segment as backchannel vs real interruption using:
      1. Duration — backchannels are short (< BACKCHANNEL_MAX_DURATION_MS).
      2. Keyword match — recognized filler/acknowledgment phrases.
    """

    def __init__(self) -> None:
        self.enabled = settings.BACKCHANNEL_DETECTION_ENABLED
        self.max_duration_ms = settings.BACKCHANNEL_MAX_DURATION_MS
        self._speech_start_time: Optional[float] = None

    def on_speech_start(self) -> None:
        """Call when VAD detects speech start."""
        self._speech_start_time = time.perf_counter()

    def on_speech_end(self) -> Optional[float]:
        """Call when VAD detects speech end.  Returns segment duration in ms."""
        if self._speech_start_time is None:
            return None
        duration_ms = (time.perf_counter() - self._speech_start_time) * 1000.0
        self._speech_start_time = None
        return duration_ms

    def classify(self, text: str, duration_ms: Optional[float] = None) -> BackchannelResult:
        """
        Classify transcribed text as backchannel or real interruption.

        Args:
            text: The transcribed speech segment.
            duration_ms: Duration of the speech segment in milliseconds.

        Returns:
            BackchannelResult with classification.
        """
        if not self.enabled:
            return BackchannelResult(is_backchannel=False, text=text, reason="detector disabled")

        normalized = text.strip().lower().rstrip(".!?,")

        # Check keyword match
        is_keyword = normalized in BACKCHANNEL_KEYWORDS

        # Check duration (short segments are more likely backchannels)
        is_short = duration_ms is not None and duration_ms < self.max_duration_ms

        if is_keyword and is_short:
            reason = f"keyword '{normalized}' + short duration ({duration_ms:.0f}ms)"
            logger.debug(f"[BackchannelDetector] Backchannel: {reason}")
            return BackchannelResult(
                is_backchannel=True, text=text, duration_ms=duration_ms, reason=reason
            )

        if is_keyword and duration_ms is None:
            # No duration info but keyword matches — treat as backchannel
            reason = f"keyword '{normalized}' (no duration info)"
            logger.debug(f"[BackchannelDetector] Backchannel: {reason}")
            return BackchannelResult(
                is_backchannel=True, text=text, duration_ms=duration_ms, reason=reason
            )

        reason = "not a backchannel"
        if is_keyword and not is_short:
            reason = f"keyword match but duration too long ({duration_ms:.0f}ms > {self.max_duration_ms}ms)"
        return BackchannelResult(
            is_backchannel=False, text=text, duration_ms=duration_ms, reason=reason
        )
