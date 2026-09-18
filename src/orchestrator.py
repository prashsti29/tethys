import logging
from typing import Optional, List

from pydantic import BaseModel, Field

from src.agents.intake import IntakeAgent
from src.agents.routing import RoutingAgent
from src.agents.scheduling import SchedulingAgent, BookingResult
from src.guardrails.red_flags import GuardrailDecision, RedFlagClassifier
from src.schemas.case_file import IntakeExtractionOutput

logger = logging.getLogger(__name__)


class OrchestratorResult(BaseModel):
    """Final structured output of a complete pipeline run."""
    # Stage 1 — Intake
    intake: Optional[IntakeExtractionOutput] = None
    # Stage 2 — Guardrails
    guardrail: Optional[GuardrailDecision] = None
    # Stage 3 — Routing
    target_specialty: Optional[str] = None
    routing_reason: Optional[str] = None
    matched_protocol: Optional[str] = None
    # Stage 4 — Scheduling
    booking: Optional[BookingResult] = None
    # Final patient-facing response
    response_text: str = Field(description="Human-readable voice response to patient")
    # Follow-up question if more info needed
    follow_up_question: Optional[str] = None
    # Whether conversation turn is complete or awaiting patient reply
    awaiting_patient_reply: bool = False


class VoiceAssistantOrchestrator:
    """
    Master orchestrator that runs the full medical intake pipeline:
      1. IntakeAgent       — extract symptoms & chief complaint
      2. RedFlagClassifier — check for medical emergencies
      3. RoutingAgent      — match symptoms to specialty via RAG
      4. SchedulingAgent   — find slots & book appointment
    """

    def __init__(self):
        self.intake_agent = IntakeAgent()
        self.guardrail = RedFlagClassifier()
        self.routing_agent = RoutingAgent()
        self.scheduling_agent = SchedulingAgent()

    def run(
        self,
        user_input: str,
        patient_name: str = "Patient",
        conversation_history: Optional[List[str]] = None,
        auto_book: bool = True,
    ) -> OrchestratorResult:
        """
        Full pipeline: text in → OrchestratorResult out.
        Set auto_book=False to skip appointment booking (for follow-up turns).
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[Orchestrator] INPUT: '{user_input}'")
        logger.info(f"{'='*60}")

        # ── Stage 1: Intake Extraction ──────────────────────────────
        logger.info("[Orchestrator] Stage 1 → Intake Agent")
        intake_result = self.intake_agent.extract_case(user_input, conversation_history)
        logger.info(
            f"  Chief Complaint: {intake_result.chief_complaint} | "
            f"Confidence: {intake_result.confidence_score} | "
            f"Symptoms: {[s.symptom_name for s in intake_result.symptoms]}"
        )

        # ── Stage 2: Guardrails (Parallel Red-Flag Check) ───────────
        logger.info("[Orchestrator] Stage 2 → Guardrail Check")
        guardrail_result = self.guardrail.evaluate(
            user_input, symptoms=intake_result.symptoms, log_to_db=True
        )

        if guardrail_result.is_red_flag:
            logger.warning(
                f"  RED FLAG: {guardrail_result.category} | Rule: {guardrail_result.rule_triggered}"
            )
            return OrchestratorResult(
                intake=intake_result,
                guardrail=guardrail_result,
                response_text=guardrail_result.emergency_instruction,
                awaiting_patient_reply=False,
            )

        # ── Stage 2b: Adaptive Follow-Up if Low Confidence ──────────
        if intake_result.confidence_score < 0.8 and intake_result.follow_up_question:
            logger.info(
                f"  Low confidence ({intake_result.confidence_score}). "
                f"Requesting follow-up: '{intake_result.follow_up_question}'"
            )
            return OrchestratorResult(
                intake=intake_result,
                guardrail=guardrail_result,
                follow_up_question=intake_result.follow_up_question,
                response_text=intake_result.follow_up_question,
                awaiting_patient_reply=True,
            )

        # ── Stage 3: RAG Routing ────────────────────────────────────
        logger.info("[Orchestrator] Stage 3 → Routing Agent")
        routing_decision = self.routing_agent.route_case(intake_result)
        logger.info(
            f"  Routed to: {routing_decision.target_specialty} | "
            f"Protocol: {routing_decision.matched_protocol_title} | "
            f"Similarity: {routing_decision.similarity_score}"
        )

        # ── Stage 4: Scheduling ─────────────────────────────────────
        if not auto_book:
            return OrchestratorResult(
                intake=intake_result,
                guardrail=guardrail_result,
                target_specialty=routing_decision.target_specialty,
                routing_reason=routing_decision.routing_reason,
                matched_protocol=routing_decision.matched_protocol_title,
                response_text=(
                    f"Based on your symptoms, I am routing you to {routing_decision.target_specialty}. "
                    f"Would you like me to check available appointments?"
                ),
                awaiting_patient_reply=True,
            )

        logger.info("[Orchestrator] Stage 4 → Scheduling Agent")
        slots = self.scheduling_agent.find_available_slots(routing_decision.target_specialty)

        if not slots:
            return OrchestratorResult(
                intake=intake_result,
                guardrail=guardrail_result,
                target_specialty=routing_decision.target_specialty,
                routing_reason=routing_decision.routing_reason,
                matched_protocol=routing_decision.matched_protocol_title,
                response_text=(
                    f"I have routed your case to {routing_decision.target_specialty}, but no "
                    f"available appointment slots were found at this time. "
                    f"Please call the clinic directly to schedule."
                ),
                awaiting_patient_reply=False,
            )

        # Book the first available slot
        first_slot = slots[0]
        booking = self.scheduling_agent.book_appointment(
            doctor_id=first_slot.doctor_id,
            slot_time=first_slot.slot_time,
            patient_name=patient_name,
        )

        response = booking.message if booking.success else (
            f"I found a {routing_decision.target_specialty} specialist but was unable to complete "
            f"the booking ({booking.message}). Please call to schedule directly."
        )

        logger.info(
            f"  Booking success={booking.success} | "
            f"Doctor: {booking.doctor_name} | Slot: {booking.slot_time}"
        )
        logger.info(f"[Orchestrator] RESPONSE: '{response}'")

        return OrchestratorResult(
            intake=intake_result,
            guardrail=guardrail_result,
            target_specialty=routing_decision.target_specialty,
            routing_reason=routing_decision.routing_reason,
            matched_protocol=routing_decision.matched_protocol_title,
            booking=booking,
            response_text=response,
            awaiting_patient_reply=False,
        )
