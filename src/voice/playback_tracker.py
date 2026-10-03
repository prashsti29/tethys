"""
Playback tracker — maps TTS audio chunks to source text spans.

On barge-in, computes the last fully-spoken word index so the LLM
conversation history can be truncated to what the patient actually heard.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TextAudioMapping:
    """One mapping entry: a text segment and its corresponding audio byte range."""
    text: str
    audio_start_byte: int
    audio_end_byte: int

    @property
    def audio_length(self) -> int:
        return self.audio_end_byte - self.audio_start_byte


class PlaybackTracker:
    """
    Tracks the relationship between synthesized text and audio bytes.

    Usage:
        1. Call ``register_segment(text, audio_bytes)`` each time TTS produces
           audio for a text chunk.
        2. Call ``update_playback_position(bytes_played)`` as audio is sent to
           the speaker.
        3. On barge-in, call ``get_spoken_text()`` to retrieve the text prefix
           that was actually played to the patient.
    """

    def __init__(self, sample_rate: int = 22050, bytes_per_sample: int = 2) -> None:
        self._mappings: List[TextAudioMapping] = []
        self._total_audio_bytes: int = 0
        self._bytes_played: int = 0
        self._sample_rate = sample_rate
        self._bytes_per_sample = bytes_per_sample

    def register_segment(self, text: str, audio_byte_count: int) -> None:
        """Register a text→audio mapping after TTS produces audio for *text*."""
        mapping = TextAudioMapping(
            text=text,
            audio_start_byte=self._total_audio_bytes,
            audio_end_byte=self._total_audio_bytes + audio_byte_count,
        )
        self._mappings.append(mapping)
        self._total_audio_bytes += audio_byte_count
        logger.debug(
            f"[PlaybackTracker] Registered: '{text[:40]}...' → "
            f"bytes [{mapping.audio_start_byte}:{mapping.audio_end_byte}]"
        )

    def update_playback_position(self, bytes_played: int) -> None:
        """Update cumulative bytes that have been sent to the speaker."""
        self._bytes_played = bytes_played

    def advance_playback(self, chunk_bytes: int) -> None:
        """Advance playback position by *chunk_bytes*."""
        self._bytes_played += chunk_bytes

    def get_spoken_text(self) -> str:
        """
        Return the text that was fully or partially played before interruption.

        Fully-played segments are included entirely. For a partially-played
        segment, words are included proportionally to bytes played.
        """
        if not self._mappings:
            return ""

        spoken_parts: List[str] = []

        for mapping in self._mappings:
            if self._bytes_played >= mapping.audio_end_byte:
                # Fully played
                spoken_parts.append(mapping.text)
            elif self._bytes_played > mapping.audio_start_byte:
                # Partially played — estimate word position
                bytes_into_segment = self._bytes_played - mapping.audio_start_byte
                fraction_played = bytes_into_segment / max(mapping.audio_length, 1)

                words = mapping.text.split()
                words_spoken = max(1, int(len(words) * fraction_played))
                spoken_parts.append(" ".join(words[:words_spoken]))
                break  # No further segments were played
            else:
                break  # Not reached yet

        return " ".join(spoken_parts).strip()

    def get_unspoken_text(self) -> str:
        """Return the text that was NOT played (complement of get_spoken_text)."""
        spoken = self.get_spoken_text()
        full_text = " ".join(m.text for m in self._mappings)
        if full_text.startswith(spoken):
            return full_text[len(spoken):].strip()
        return ""

    def reset(self) -> None:
        """Clear all mappings and playback state for the next turn."""
        self._mappings.clear()
        self._total_audio_bytes = 0
        self._bytes_played = 0

    @property
    def total_audio_bytes(self) -> int:
        return self._total_audio_bytes

    @property
    def bytes_played(self) -> int:
        return self._bytes_played

    @property
    def playback_fraction(self) -> float:
        if self._total_audio_bytes == 0:
            return 0.0
        return min(1.0, self._bytes_played / self._total_audio_bytes)
