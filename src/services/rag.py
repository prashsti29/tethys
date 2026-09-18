import logging
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from src.db.session import get_db_session
from src.models.entities import ProtocolDoc

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG service embedding protocol documents into 384-dimensional vectors
    and executing pgvector similarity searches for clinical triage routing.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"[RAGService] Loading SentenceTransformer model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def generate_embedding(self, text: str) -> List[float]:
        """Encodes text into a 384-dimensional float vector."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def index_protocols(self) -> int:
        """
        Computes vector embeddings for all ProtocolDoc records in Postgres
        and updates their embedding column.
        """
        logger.info("[RAGService] Indexing protocol documents into pgvector...")
        indexed_count = 0

        with get_db_session() as db:
            docs = db.query(ProtocolDoc).all()
            for doc in docs:
                text_to_embed = f"Title: {doc.title}. Specialty: {doc.specialty}. Content: {doc.content}"
                embedding = self.generate_embedding(text_to_embed)
                doc.embedding = embedding
                indexed_count += 1

            db.commit()
            logger.info(f"[RAGService] Successfully indexed {indexed_count} protocol documents into pgvector.")

        return indexed_count

    def search_protocols(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Executes vector cosine similarity search in Postgres pgvector.
        Returns matched documents sorted by highest similarity score.
        """
        query_vector = self.generate_embedding(query)
        results = []

        with get_db_session() as db:
            # pgvector cosine distance: ProtocolDoc.embedding.cosine_distance(query_vector)
            # Similarity score = 1.0 - distance
            distance_expr = ProtocolDoc.embedding.cosine_distance(query_vector)
            stmt = select(ProtocolDoc, distance_expr.label("distance")).order_by("distance").limit(limit)

            rows = db.execute(stmt).all()
            for doc, distance in rows:
                similarity = max(0.0, min(1.0, round(1.0 - float(distance), 4)))
                results.append({
                    "id": doc.id,
                    "title": doc.title,
                    "specialty": doc.specialty,
                    "content": doc.content,
                    "distance": round(float(distance), 4),
                    "similarity": similarity,
                })

        return results
