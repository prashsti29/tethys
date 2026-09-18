import logging
from typing import Optional
from pydantic import BaseModel, Field

from src.schemas.case_file import IntakeExtractionOutput
from src.services.rag import RAGService

logger = logging.getLogger(__name__)


class RoutingDecision(BaseModel):
    target_specialty: str = Field(description="Target medical specialty (Cardiology, Neurology, Primary Care, Dermatology, Orthopedics)")
    matched_protocol_title: Optional[str] = Field(default=None, description="Title of matched medical protocol document")
    similarity_score: float = Field(default=0.0, description="RAG vector similarity score")
    routing_reason: str = Field(description="Explanation of routing decision")


SPECIALTY_MAPPINGS = {
    "cardiology": "Cardiology",
    "neurology": "Neurology",
    "dermatology": "Dermatology",
    "orthopedics": "Orthopedics",
    "primary care": "Primary Care",
}


class RoutingAgent:
    """
    Routing Agent evaluating intake extractions against vector-indexed medical protocols
    and assigning patient cases to appropriate medical specialties.
    """

    def __init__(self, rag_service: Optional[RAGService] = None):
        self.rag_service = rag_service or RAGService()

    def route_case(self, case_data: IntakeExtractionOutput) -> RoutingDecision:
        """
        Routes intake case to matching medical specialty based on RAG similarity search.
        """
        # Construct search query from chief complaint & symptoms
        query_parts = [case_data.chief_complaint]
        for sym in case_data.symptoms:
            query_parts.append(f"{sym.symptom_name} {sym.body_location or ''}")

        search_query = " ".join(query_parts).strip()
        logger.info(f"[RoutingAgent] Executing RAG routing for query: '{search_query}'")

        matches = self.rag_service.search_protocols(search_query, limit=3)
        if not matches:
            logger.info("[RoutingAgent] No protocols found in pgvector. Defaulting to Primary Care.")
            return RoutingDecision(
                target_specialty="Primary Care",
                matched_protocol_title=None,
                similarity_score=0.0,
                routing_reason="No protocol matches found in vector index. Defaulted to Primary Care.",
            )

        best_match = matches[0]
        similarity = best_match["similarity"]
        raw_specialty = best_match["specialty"].lower()

        logger.info(
            f"[RoutingAgent] Best match protocol: '{best_match['title']}' | "
            f"Raw Specialty: '{best_match['specialty']}' | Similarity: {similarity}"
        )

        if similarity >= 0.35:
            # Map raw specialty string to clean target specialty category
            target_spec = "Primary Care"
            for key, val in SPECIALTY_MAPPINGS.items():
                if key in raw_specialty:
                    target_spec = val
                    break

            return RoutingDecision(
                target_specialty=target_spec,
                matched_protocol_title=best_match["title"],
                similarity_score=similarity,
                routing_reason=f"Matched protocol '{best_match['title']}' with similarity {similarity}.",
            )
        else:
            logger.info(f"[RoutingAgent] Similarity {similarity} below threshold 0.35. Defaulting to Primary Care.")
            return RoutingDecision(
                target_specialty="Primary Care",
                matched_protocol_title=best_match["title"],
                similarity_score=similarity,
                routing_reason="Protocol similarity below confidence threshold. Routed to Primary Care for general assessment.",
            )
