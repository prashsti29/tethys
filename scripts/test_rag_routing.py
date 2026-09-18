#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.routing import RoutingAgent
from src.schemas.case_file import IntakeExtractionOutput, SymptomExtraction
from src.services.rag import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_rag_routing")


def main():
    logger.info("=== Starting Phase 5 RAG & Routing Agent Evaluation ===")

    # 1. Index Protocol Embeddings in Postgres pgvector
    rag_service = RAGService()
    indexed_count = rag_service.index_protocols()
    logger.info(f"Indexed {indexed_count} protocols in pgvector database.")

    # 2. Evaluate Routing Agent Across Clinical Test Cases
    routing_agent = RoutingAgent(rag_service=rag_service)

    test_cases = [
        {
            "name": "Case 1: Cardiology Symptoms",
            "extraction": IntakeExtractionOutput(
                chief_complaint="Chest pain radiating to left arm",
                symptoms=[SymptomExtraction(symptom_name="Chest pain", body_location="Left chest")],
                confidence_score=0.9,
            ),
            "expected_specialty": "Cardiology",
        },
        {
            "name": "Case 2: Neurology Symptoms",
            "extraction": IntakeExtractionOutput(
                chief_complaint="Sudden severe headache with dizziness",
                symptoms=[SymptomExtraction(symptom_name="Headache", body_location="Head")],
                confidence_score=0.85,
            ),
            "expected_specialty": "Neurology",
        },
        {
            "name": "Case 3: Dermatology Symptoms",
            "extraction": IntakeExtractionOutput(
                chief_complaint="Localized itchy skin rash on arm",
                symptoms=[SymptomExtraction(symptom_name="Rash", body_location="Arm")],
                confidence_score=0.88,
            ),
            "expected_specialty": "Dermatology",
        },
        {
            "name": "Case 4: Primary Care Symptoms",
            "extraction": IntakeExtractionOutput(
                chief_complaint="Mild fatigue and seasonal cough for 2 days",
                symptoms=[SymptomExtraction(symptom_name="Cough", duration="2 days")],
                confidence_score=0.8,
            ),
            "expected_specialty": "Primary Care",
        },
    ]

    for tc in test_cases:
        logger.info(f"\n--- Running {tc['name']} ---")
        decision = routing_agent.route_case(tc["extraction"])

        logger.info(f"Target Specialty : {decision.target_specialty}")
        logger.info(f"Matched Protocol : {decision.matched_protocol_title}")
        logger.info(f"Similarity Score : {decision.similarity_score}")
        logger.info(f"Routing Reason   : {decision.routing_reason}")

        assert decision.target_specialty == tc["expected_specialty"], (
            f"ERROR: Expected specialty {tc['expected_specialty']} but got {decision.target_specialty}"
        )

    logger.info("\n=== RAG & Routing Agent Evaluation Completed Successfully! All Specialty Matches Verified ===")


if __name__ == "__main__":
    main()
