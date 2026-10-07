import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationSessionState:
    session_id: str
    patient_name: str = "Patient"
    conversation_history: List[str] = field(default_factory=list)
    audio_offset_ms: float = 0.0
    active_agent: str = "intake"
    awaiting_patient_reply: bool = False
    metadata: Dict = field(default_factory=dict)


class SessionStore:
    """
    Manages mid-call session state persistence and re-hydration to survive context loss,
    worker crashes, and mid-call reconnections.
    """

    def __init__(self, storage_dir: str = "/tmp/voice_assistant_sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def save_session(self, state: ConversationSessionState) -> None:
        """Serializes and persists active conversation state."""
        try:
            path = self._get_path(state.session_id)
            data = asdict(state)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"[SessionStore] Saved session state: {state.session_id}")
        except Exception as e:
            logger.error(f"[SessionStore] Error saving session {state.session_id}: {e}")

    def load_session(self, session_id: str) -> Optional[ConversationSessionState]:
        """Loads and re-hydrates conversation state for a reconnecting call."""
        path = self._get_path(session_id)
        if not path.exists():
            logger.info(f"[SessionStore] No saved state found for session: {session_id}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = ConversationSessionState(**data)
            logger.info(f"[SessionStore] Successfully re-hydrated session: {session_id} ({len(state.conversation_history)} turns)")
            return state
        except Exception as e:
            logger.error(f"[SessionStore] Error loading session {session_id}: {e}")
            return None

    def delete_session(self, session_id: str) -> None:
        """Removes session state when call finishes."""
        path = self._get_path(session_id)
        if path.exists():
            path.unlink()
            logger.info(f"[SessionStore] Removed session state: {session_id}")
