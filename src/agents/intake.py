import json
import logging
import re
from typing import Dict, List, Optional
import httpx

from src.config import settings
from src.schemas.case_file import IntakeExtractionOutput, SymptomExtraction

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert clinical medical intake extraction assistant.
Analyze the patient's spoken audio transcript or text input and extract clinical information into structured JSON.

Return ONLY a raw JSON object matching the following exact JSON schema keys:
{
  "chief_complaint": "primary reason for seeking medical care",
  "symptoms": [
    {
      "symptom_name": "name of symptom",
      "duration": "duration if stated (e.g. 2 hours, 3 days) or null",
      "severity": 1 to 10 integer rating if stated or null,
      "body_location": "anatomical location or null"
    }
  ],
  "confidence_score": float between 0.0 and 1.0,
  "follow_up_question": "one concise follow-up question if critical information like duration or severity is missing, or null",
  "red_flag_detected": true or false,
  "red_flag_reason": "explanation if emergency red flag detected (e.g. severe radiating chest pain, sudden thunderclap headache) or null"
}
Do NOT include markdown wrapping or extra commentary outside the JSON block.
"""


class IntakeAgent:
    """
    Intake Agent extracting clinical symptoms, computing confidence scores,
    generating adaptive follow-up questions, and providing a resilient fallback.
    """

    def __init__(self, ollama_host: str = None, model_name: str = None):
        self.ollama_host = ollama_host or settings.OLLAMA_HOST
        self.model_name = model_name or settings.OLLAMA_MODEL

    def extract_case(self, user_input: str, conversation_history: Optional[List[str]] = None) -> IntakeExtractionOutput:
        """
        Main entry point for extracting clinical case information from user transcript.
        Tries Ollama LLM first; falls back to rule-based extractor if offline or invalid.
        """
        logger.info(f"[IntakeAgent] Processing input: '{user_input}'")

        try:
            extraction = self._call_ollama(user_input, conversation_history)
            if extraction:
                logger.info("[IntakeAgent] Successfully extracted case via Ollama LLM.")
                return self._enrich_confidence_and_followup(extraction, user_input)
        except Exception as e:
            logger.warning(f"[IntakeAgent] Ollama LLM call failed or unavailable: {e}. Executing fallback extractor.")

        # Fallback to rule-based regex extractor
        logger.info("[IntakeAgent] Executing rule-based fallback extractor.")
        fallback_res = self._fallback_extraction(user_input)
        return self._enrich_confidence_and_followup(fallback_res, user_input)

    def _call_ollama(self, user_input: str, conversation_history: Optional[List[str]]) -> Optional[IntakeExtractionOutput]:
        context_str = ""
        if conversation_history:
            context_str = "\nPrior conversation:\n" + "\n".join(conversation_history)

        prompt = f"{context_str}\nPatient Transcript: \"{user_input}\""

        url = f"{self.ollama_host.rstrip('/')}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1}
        }

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "").strip()
        if not content:
            return None

        # Parse JSON and validate with Pydantic
        clean_json = self._clean_json_string(content)
        parsed_dict = json.loads(clean_json)
        return IntakeExtractionOutput.model_validate(parsed_dict)

    def _clean_json_string(self, text: str) -> str:
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _fallback_extraction(self, text: str) -> IntakeExtractionOutput:
        """
        Hardcoded rule-based regex fallback extractor.
        Guarantees zero-crash intake parsing even when LLM is offline.
        """
        text_lower = text.lower()
        symptoms: List[SymptomExtraction] = []
        red_flag = False
        red_flag_reason = None

        # Red flag keyword checks
        if any(rf in text_lower for rf in ["radiating chest pain", "chest pain left arm", "thunderclap headache", "passed out"]):
            red_flag = True
            red_flag_reason = "Emergency Red Flag: Potential acute cardiovascular or neurological event detected."
        elif "chest pain" in text_lower and any(kw in text_lower for kw in ["severe", "crushing", "arm"]):
            red_flag = True
            red_flag_reason = "Emergency Red Flag: Severe chest pain symptoms detected."

        # Symptom extraction rules
        known_symptoms = [
            ("chest pain", ["chest"]),
            ("headache", ["head"]),
            ("fever", ["whole body"]),
            ("cough", ["throat", "chest"]),
            ("rash", ["skin"]),
            ("dizziness", ["head"]),
            ("nausea", ["stomach"]),
            ("shortness of breath", ["chest", "lungs"]),
            ("stomach ache", ["abdomen"]),
        ]

        for sym_name, default_loc in known_symptoms:
            if sym_name in text_lower:
                # Extract duration pattern e.g. "3 days", "2 hours"
                dur_match = re.search(r"(\d+\s*(?:days?|hours?|weeks?|months?|mins?|minutes?))", text_lower)
                duration = dur_match.group(1) if dur_match else None

                # Extract severity pattern e.g. "7 out of 10", "8/10", "severity 9"
                sev_match = re.search(r"(?:severity\s*(\d{1,2})|(\d{1,2})\s*(?:out of 10|\/10))", text_lower)
                severity = None
                if sev_match:
                    sev_val = int(sev_match.group(1) or sev_match.group(2))
                    if 1 <= sev_val <= 10:
                        severity = sev_val

                symptoms.append(
                    SymptomExtraction(
                        symptom_name=sym_name.capitalize(),
                        duration=duration,
                        severity=severity,
                        body_location=default_loc[0] if default_loc else None,
                    )
                )

        chief_complaint = text.strip()
        if symptoms:
            chief_complaint = f"Patient reports {', '.join([s.symptom_name for s in symptoms])}."

        return IntakeExtractionOutput(
            chief_complaint=chief_complaint,
            symptoms=symptoms if symptoms else [SymptomExtraction(symptom_name=text[:50])],
            confidence_score=0.5,
            red_flag_detected=red_flag,
            red_flag_reason=red_flag_reason,
        )

    def _enrich_confidence_and_followup(self, extraction: IntakeExtractionOutput, user_input: str) -> IntakeExtractionOutput:
        """
        Calculates confidence score and generates ONE adaptive follow-up question if details are missing.
        """
        score = 0.4

        if extraction.chief_complaint and len(extraction.chief_complaint) > 5:
            score += 0.2

        if extraction.symptoms:
            score += 0.1
            has_duration = any(s.duration is not None for s in extraction.symptoms)
            has_severity = any(s.severity is not None for s in extraction.symptoms)

            if has_duration:
                score += 0.15
            if has_severity:
                score += 0.15

        score = min(1.0, round(score, 2))
        extraction.confidence_score = score

        # Generate single adaptive follow-up question if confidence < 0.8
        if score < 0.8 and not extraction.red_flag_detected:
            has_dur = any(s.duration is not None for s in extraction.symptoms)
            has_sev = any(s.severity is not None for s in extraction.symptoms)

            if not has_dur:
                extraction.follow_up_question = "How long have you been experiencing these symptoms?"
            elif not has_sev:
                extraction.follow_up_question = "On a scale of 1 to 10, how severe is your pain or discomfort?"
            else:
                extraction.follow_up_question = "Could you tell me a bit more about where exactly you feel the discomfort?"
        else:
            extraction.follow_up_question = None

        return extraction
