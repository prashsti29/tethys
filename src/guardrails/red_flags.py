import logging
import re
from typing import List, Optional
from pydantic import BaseModel, Field

from src.db.session import get_db_session
from src.models.entities import AuditLog
from src.schemas.case_file import SymptomExtraction

logger = logging.getLogger(__name__)


class GuardrailDecision(BaseModel):
    is_red_flag: bool = Field(default=False, description="True if an emergency red flag is triggered")
    category: Optional[str] = Field(default=None, description="Medical category (CARDIOVASCULAR, NEUROLOGICAL, RESPIRATORY, ANAPHYLAXIS, TRAUMA)")
    emergency_instruction: Optional[str] = Field(default=None, description="Direct ER / 911 instruction given to patient")
    allow_scheduling: bool = Field(default=True, description="False if red flag prevents routine scheduling path")
    flag_human_review: bool = Field(default=False, description="True if case must be routed for human clinician review")
    rule_triggered: Optional[str] = Field(default=None, description="Clinical pattern rule matched")


CLINICAL_RULES = [
    {
        "category": "CARDIOVASCULAR",
        "patterns": [
            r"radiating\s+chest\s+pain",
            r"chest\s+pain\s+.*left\s+arm",
            r"chest\s+pain\s+.*jaw",
            r"crushing\s+chest\s+pressure",
            r"chest\s+pain\s+.*cold\s+sweat",
            r"diaphoresis",
            r"chest\s+tightness\s+.*shortness\s+of\s+breath",
        ],
        "instruction": (
            "EMERGENCY ALERT: Your symptoms suggest a potential acute cardiovascular emergency. "
            "Stop intake immediately and call 911 or go to the nearest Emergency Room. Do not wait for a clinic appointment."
        ),
    },
    {
        "category": "NEUROLOGICAL",
        "patterns": [
            r"thunderclap\s+headache",
            r"worst\s+headache\s+of\s+my\s+life",
            r"facial\s+droop",
            r"arm\s+weakness\s+.*slurred\s+speech",
            r"sudden\s+numbness\s+.*one\s+side",
            r"sudden\s+loss\s+of\s+vision",
            r"sudden\s+confusion",
        ],
        "instruction": (
            "EMERGENCY ALERT: Your symptoms suggest a potential acute neurological emergency (such as a stroke or hemorrhage). "
            "Call 911 or proceed to the nearest Emergency Department immediately."
        ),
    },
    {
        "category": "RESPIRATORY",
        "patterns": [
            r"severe\s+shortness\s+of\s+breath",
            r"gasping\s+for\s+air",
            r"unable\s+to\s+speak\s+full\s+sentences",
            r"lips?\s+turning\s+blue",
            r"cyanosis",
            r"stridor",
        ],
        "instruction": (
            "EMERGENCY ALERT: Severe respiratory distress detected. Seek immediate emergency medical care (call 911)."
        ),
    },
    {
        "category": "ANAPHYLAXIS",
        "patterns": [
            r"swelling\s+of\s+lips",
            r"throat\s+closing\s+up",
            r"anaphylaxis",
            r"difficulty\s+breathing\s+after\s+(?:bee\s+sting|allergy|food)",
        ],
        "instruction": (
            "EMERGENCY ALERT: Potential severe allergic reaction (anaphylaxis). Use an EpiPen if available and call 911 immediately."
        ),
    },
    {
        "category": "TRAUMA",
        "patterns": [
            r"uncontrolled\s+bleeding",
            r"head\s+trauma\s+.*loss\s+of\s+consciousness",
            r"severe\s+open\s+fracture",
        ],
        "instruction": (
            "EMERGENCY ALERT: Severe physical trauma detected. Seek emergency department care immediately."
        ),
    },
]


class RedFlagClassifier:
    """
    Rule-based red-flag classifier running parallel to intake LLM.
    Guarantees immediate emergency triage and blocks routine scheduling paths.
    """

    def evaluate(
        self,
        text: str,
        symptoms: Optional[List[SymptomExtraction]] = None,
        log_to_db: bool = True
    ) -> GuardrailDecision:
        logger.info(f"[RedFlagClassifier] Evaluating text: '{text}'")

        text_to_check = text.lower()
        if symptoms:
            sym_text = " ".join([f"{s.symptom_name} {s.body_location or ''}" for s in symptoms]).lower()
            text_to_check += " " + sym_text

        for rule in CLINICAL_RULES:
            category = rule["category"]
            instruction = rule["instruction"]

            for pattern in rule["patterns"]:
                if re.search(pattern, text_to_check, re.IGNORECASE):
                    matched_str = pattern
                    logger.warning(
                        f"[RedFlagClassifier] RED FLAG TRIGGERED! Category={category}, Pattern='{matched_str}'"
                    )

                    decision = GuardrailDecision(
                        is_red_flag=True,
                        category=category,
                        emergency_instruction=instruction,
                        allow_scheduling=False,
                        flag_human_review=True,
                        rule_triggered=matched_str,
                    )

                    if log_to_db:
                        self._log_audit_event(decision, text)

                    return decision

        logger.info("[RedFlagClassifier] No red flags detected. Routine care path allowed.")
        return GuardrailDecision(
            is_red_flag=False,
            category=None,
            emergency_instruction=None,
            allow_scheduling=True,
            flag_human_review=False,
            rule_triggered=None,
        )

    def _log_audit_event(self, decision: GuardrailDecision, user_input: str):
        try:
            with get_db_session() as db:
                audit = AuditLog(
                    event_type="RED_FLAG_TRIGGERED",
                    agent_name="GuardrailEngine",
                    payload={
                        "category": decision.category,
                        "rule_triggered": decision.rule_triggered,
                        "emergency_instruction": decision.emergency_instruction,
                        "allow_scheduling": decision.allow_scheduling,
                        "user_input": user_input[:200],
                    },
                )
                db.add(audit)
            logger.info("[RedFlagClassifier] Successfully persisted red-flag audit record to Postgres.")
        except Exception as e:
            logger.error(f"[RedFlagClassifier] Failed to persist audit log: {e}")
