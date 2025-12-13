"""
OSHA Standards RAG System.

Retrieval-Augmented Generation for OSHA safety regulations.
"""

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import structlog

log = structlog.get_logger()

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class OshaChunk:
    """A chunk of OSHA regulation text."""
    chunk_id: str
    text: str
    section: str  # "1910.212"
    subsection: str  # "(a)(1)"
    title: str  # "Point of Operation Guarding"
    source_file: str
    embedding: Optional[list[float]] = None


@dataclass
class SearchResult:
    """Result from RAG query."""
    text: str
    section: str
    title: str
    score: float
    subsection: str = ""


# Key OSHA 1910 sections for factory safety
OSHA_SECTIONS = {
    "1910.132": {
        "title": "PPE General Requirements",
        "text": """Employers must assess the workplace to determine if hazards are present that require the use of personal protective equipment (PPE). If such hazards exist, the employer must select appropriate PPE, provide it to employees, and ensure its use. PPE must be properly maintained and replaced when necessary. Training on proper use is required."""
    },
    "1910.133": {
        "title": "Eye and Face Protection",
        "text": """Employers must ensure employees use appropriate eye or face protection when exposed to eye or face hazards from flying particles, molten metal, liquid chemicals, acids or caustic liquids, chemical gases or vapors, or potentially injurious light radiation. Side protection is required when there is a hazard from flying objects. Detachable side protectors are acceptable."""
    },
    "1910.135": {
        "title": "Head Protection",
        "text": """Employees working in areas where there is a possible danger of head injury from impact, falling or flying objects, or electrical shock and burns must be protected by protective helmets. Helmets must comply with ANSI Z89.1 standards. Helmets must be maintained in good condition and replaced if damaged."""
    },
    "1910.136": {
        "title": "Foot Protection",
        "text": """Employees must use protective footwear when working in areas where there is a danger of foot injuries from falling or rolling objects, objects piercing the sole, or electrical hazards. Safety-toe footwear must meet ANSI Z41 requirements. Employers must ensure proper fit and condition of protective footwear."""
    },
    "1910.138": {
        "title": "Hand Protection",
        "text": """Employers must select and require employees to use appropriate hand protection when hands are exposed to hazards such as skin absorption of harmful substances, severe cuts or lacerations, severe abrasions, punctures, chemical burns, thermal burns, and harmful temperature extremes. Selection must be based on performance characteristics relative to the task."""
    },
    "1910.212": {
        "title": "Machine Guarding",
        "text": """One or more methods of machine guarding must protect operators and other employees from hazards such as point of operation, ingoing nip points, rotating parts, flying chips and sparks. Guards must be affixed to the machine where possible. Point of operation guards must prevent the operator from having any part of their body in the danger zone during the operating cycle."""
    },
    "1910.147": {
        "title": "Lockout/Tagout (LOTO)",
        "text": """This standard establishes minimum performance requirements for control of hazardous energy during servicing and maintenance. Energy sources include electrical, mechanical, hydraulic, pneumatic, chemical, thermal, and other sources. Lockout devices must be durable, standardized, substantial, and identifiable. Procedures must be developed and documented for each machine or equipment."""
    },
    "1910.252": {
        "title": "Welding, Cutting and Brazing - General Requirements",
        "text": """Fire prevention and protection measures are required for all welding and cutting operations. Workers performing welding must be provided with proper PPE including helmets with filter lenses, protective clothing, and gloves. Adequate ventilation must be provided. Work permits may be required for hot work in certain areas. Fire watch may be required for 30 minutes after welding operations."""
    },
    "1910.134": {
        "title": "Respiratory Protection",
        "text": """Employers must provide respirators when such equipment is necessary to protect employee health. A written respiratory protection program is required with procedures for selection, medical evaluations, fit testing, use, and maintenance. Employees must be trained on respirator use, limitations, and proper maintenance."""
    },
    "1910.95": {
        "title": "Occupational Noise Exposure",
        "text": """When employees are exposed to noise levels at or above 85 decibels averaged over 8 hours, employers must implement a hearing conservation program. This includes monitoring, audiometric testing, hearing protectors, training, and recordkeeping. Hearing protection must attenuate noise to acceptable levels."""
    }
}


class OshaRAG:
    """
    RAG system for OSHA regulation lookup.

    Features:
    - Semantic search for relevant regulations
    - Direct reference lookup
    - Citation formatting
    - FAISS vector index
    """

    def __init__(
        self,
        docs_path: str = "data/osha_standards",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        index_path: str = "data/osha_index"
    ):
        """
        Initialize OSHA RAG system.

        Args:
            docs_path: Path to OSHA documents
            embedding_model: Sentence transformer model for embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            index_path: Path to save/load index
        """
        self.docs_path = Path(docs_path)
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = Path(index_path)

        self._chunks: list[OshaChunk] = []
        self._embeddings: Optional[object] = None  # numpy array
        self._index: Optional[object] = None  # FAISS index
        self._encoder: Optional[SentenceTransformer] = None
        self._initialized = False

        log.info(
            "osha_rag_init",
            docs_path=str(docs_path),
            embedding_model=embedding_model
        )

    def _load_encoder(self) -> None:
        """Load the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")

        if self._encoder is None:
            log.info("loading_embedding_model", model=self.embedding_model_name)
            self._encoder = SentenceTransformer(self.embedding_model_name)

    def _chunk_text(self, text: str, section: str, title: str, source: str) -> list[OshaChunk]:
        """Split text into chunks."""
        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            # Generate chunk ID
            chunk_id = hashlib.md5(f"{section}_{i}_{chunk_text[:50]}".encode()).hexdigest()[:12]

            chunks.append(OshaChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                section=section,
                subsection="",
                title=title,
                source_file=source
            ))

        return chunks

    def build_index(self) -> None:
        """Build or rebuild the vector index."""
        self._load_encoder()

        # Load built-in OSHA sections
        self._chunks = []

        for section, data in OSHA_SECTIONS.items():
            chunks = self._chunk_text(
                data["text"],
                section,
                data["title"],
                "built_in"
            )
            self._chunks.extend(chunks)

        # Try to load additional documents from docs_path
        if self.docs_path.exists():
            for file_path in self.docs_path.glob("*.txt"):
                try:
                    with open(file_path, "r") as f:
                        text = f.read()

                    # Try to extract section from filename
                    section = file_path.stem
                    chunks = self._chunk_text(text, section, section, str(file_path))
                    self._chunks.extend(chunks)
                except Exception as e:
                    log.warning("failed_to_load_document", path=str(file_path), error=str(e))

        if not self._chunks:
            log.warning("no_chunks_created")
            return

        # Generate embeddings
        log.info("generating_embeddings", num_chunks=len(self._chunks))
        texts = [chunk.text for chunk in self._chunks]
        embeddings = self._encoder.encode(texts, show_progress_bar=True)

        # Store embeddings
        import numpy as np
        self._embeddings = np.array(embeddings).astype('float32')

        # Build FAISS index
        if FAISS_AVAILABLE:
            dimension = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            faiss.normalize_L2(self._embeddings)  # Normalize for cosine similarity
            self._index.add(self._embeddings)
            log.info("faiss_index_built", vectors=self._index.ntotal)

        self._initialized = True

        # Save index
        self._save_index()

    def _save_index(self) -> None:
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save chunks
        chunks_path = self.index_path / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)

        # Save embeddings
        if self._embeddings is not None:
            import numpy as np
            emb_path = self.index_path / "embeddings.npy"
            np.save(emb_path, self._embeddings)

        # Save FAISS index
        if FAISS_AVAILABLE and self._index is not None:
            index_path = self.index_path / "faiss.index"
            faiss.write_index(self._index, str(index_path))

        log.info("index_saved", path=str(self.index_path))

    def _load_index(self) -> bool:
        """Load index from disk."""
        chunks_path = self.index_path / "chunks.pkl"
        emb_path = self.index_path / "embeddings.npy"
        faiss_path = self.index_path / "faiss.index"

        if not chunks_path.exists():
            return False

        try:
            # Load chunks
            with open(chunks_path, "rb") as f:
                self._chunks = pickle.load(f)

            # Load embeddings
            if emb_path.exists():
                import numpy as np
                self._embeddings = np.load(emb_path)

            # Load FAISS index
            if FAISS_AVAILABLE and faiss_path.exists():
                self._index = faiss.read_index(str(faiss_path))

            self._initialized = True
            log.info("index_loaded", path=str(self.index_path), chunks=len(self._chunks))
            return True

        except Exception as e:
            log.warning("index_load_failed", error=str(e))
            return False

    def query(
        self,
        query: str,
        top_k: int = 3
    ) -> list[SearchResult]:
        """
        Query for relevant OSHA regulations.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant regulation excerpts
        """
        # Try to load cached index
        if not self._initialized:
            if not self._load_index():
                self.build_index()

        self._load_encoder()

        if not self._chunks:
            return []

        # Encode query
        import numpy as np
        query_embedding = self._encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        if FAISS_AVAILABLE and self._index is not None:
            faiss.normalize_L2(query_embedding)
            scores, indices = self._index.search(query_embedding, top_k)

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self._chunks):
                    chunk = self._chunks[idx]
                    results.append(SearchResult(
                        text=chunk.text,
                        section=chunk.section,
                        title=chunk.title,
                        score=float(score),
                        subsection=chunk.subsection
                    ))

            return results
        else:
            # Fallback: brute force search
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(query_embedding, self._embeddings)[0]
            top_indices = similarities.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                chunk = self._chunks[idx]
                results.append(SearchResult(
                    text=chunk.text,
                    section=chunk.section,
                    title=chunk.title,
                    score=float(similarities[idx]),
                    subsection=chunk.subsection
                ))

            return results

    def get_regulation(self, ref: str) -> Optional[str]:
        """
        Direct lookup of OSHA regulation by reference.

        Args:
            ref: OSHA reference (e.g., "1910.212")

        Returns:
            Regulation text if found
        """
        # Normalize reference
        ref_normalized = ref.replace("29 CFR ", "").strip()

        # Check built-in sections
        if ref_normalized in OSHA_SECTIONS:
            section = OSHA_SECTIONS[ref_normalized]
            return f"**{ref_normalized} - {section['title']}**\n\n{section['text']}"

        # Search in chunks
        for chunk in self._chunks:
            if chunk.section == ref_normalized:
                return f"**{chunk.section} - {chunk.title}**\n\n{chunk.text}"

        return None

    def get_ppe_requirements(self, hazard_type: str) -> list[SearchResult]:
        """
        Get PPE requirements for a specific hazard type.

        Args:
            hazard_type: Type of hazard (e.g., "welding", "chemical", "noise")

        Returns:
            Relevant PPE regulations
        """
        query = f"PPE requirements for {hazard_type} hazards protective equipment"
        return self.query(query, top_k=3)

    def format_citation(self, result: SearchResult) -> str:
        """Format a search result as an OSHA citation."""
        return f"29 CFR {result.section}: {result.title}"

    def get_all_sections(self) -> dict[str, str]:
        """Get all available OSHA sections."""
        return {section: data["title"] for section, data in OSHA_SECTIONS.items()}


# Convenience function
def get_osha_context(
    violation_type: str,
    zone_type: str = ""
) -> str:
    """
    Get relevant OSHA context for a violation.

    Args:
        violation_type: Type of violation (e.g., "missing_ppe")
        zone_type: Type of zone (e.g., "welding", "machine")

    Returns:
        Relevant OSHA context string
    """
    rag = OshaRAG()

    if violation_type == "missing_ppe":
        if "welding" in zone_type.lower():
            results = rag.query("welding PPE requirements eye face protection", top_k=2)
        elif "machine" in zone_type.lower():
            results = rag.query("machine guarding PPE requirements", top_k=2)
        else:
            results = rag.query("general PPE requirements workplace", top_k=2)
    elif violation_type == "restricted_area":
        results = rag.query("restricted area access control hazardous", top_k=2)
    else:
        results = rag.query("workplace safety requirements general", top_k=2)

    if results:
        context_parts = []
        for r in results:
            context_parts.append(f"[{r.section}] {r.text[:200]}...")
        return "\n\n".join(context_parts)

    return "General OSHA workplace safety requirements apply."
