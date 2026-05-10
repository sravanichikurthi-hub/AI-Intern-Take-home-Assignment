"""
retrieval.py — FAISS-based semantic retrieval over the SHL catalog.

The index is built once at startup and held in memory.
Every search call is synchronous and takes < 10 ms for a catalog of ~400 items.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Try to use sentence-transformers (better semantic matching).
# Fall back to TF-IDF if the model isn't downloadable (sandboxed/offline env).
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    _USE_SBERT = True
    _MODEL_NAME = "all-MiniLM-L6-v2"
    log.info("Will use sentence-transformers + FAISS for retrieval.")
except Exception:
    _USE_SBERT = False
    log.info("sentence-transformers unavailable; will use TF-IDF + cosine similarity.")


class CatalogRetriever:
    """
    Wraps the SHL catalog with FAISS vector search.

    Each catalog item is encoded as:
        "{name}. {description}. Keywords: {keywords}. Type: {test_type_label}."

    This dense-text representation lets the model match role descriptions
    ("senior Java developer who works with stakeholders") to the right tests.
    """

    TEST_TYPE_LABELS = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations",
    }

    def __init__(self, catalog_path: str | Path) -> None:
        catalog_path = Path(catalog_path)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")

        self.catalog: list[dict] = json.loads(catalog_path.read_text(encoding="utf-8"))
        log.info("Loaded %d assessments from catalog.", len(self.catalog))

        self._texts = [self._item_to_text(item) for item in self.catalog]

        if _USE_SBERT:
            self._setup_sbert()
        else:
            self._setup_tfidf()

    def _setup_sbert(self) -> None:
        try:
            self._model = SentenceTransformer(_MODEL_NAME)
            embeddings = self._model.encode(self._texts, show_progress_bar=False, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)
            self._index = faiss.IndexFlatIP(embeddings.shape[1])
            self._index.add(embeddings)
            self._backend = "sbert"
            log.info("FAISS index ready (%d vectors, dim=%d).", len(self.catalog), self._index.d)
        except Exception as exc:
            log.warning("SBERT setup failed (%s); falling back to TF-IDF.", exc)
            self._setup_tfidf()

    def _setup_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._tfidf_matrix = self._tfidf.fit_transform(self._texts)
        self._backend = "tfidf"
        log.info("TF-IDF index ready (%d documents).", len(self.catalog))

    def _item_to_text(self, item: dict) -> str:
        """Build a rich text representation of one catalog entry."""
        parts = [item.get("name", "")]

        desc = item.get("description", "")
        if desc:
            parts.append(desc)

        kws = item.get("keywords", [])
        if kws:
            parts.append("Keywords: " + ", ".join(kws))

        types = item.get("test_type", "")
        type_labels = []
        for code in types.split():
            label = self.TEST_TYPE_LABELS.get(code.strip(), code)
            type_labels.append(label)
        if type_labels:
            parts.append("Test type: " + ", ".join(type_labels))

        levels = item.get("job_levels", [])
        if levels:
            parts.append("Job levels: " + ", ".join(levels))

        return ". ".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 15,
        test_type_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return up to k catalog items most semantically similar to query."""
        if self._backend == "sbert":
            results = self._search_sbert(query, k * 3)
        else:
            results = self._search_tfidf(query, k * 3)

        if test_type_filter:
            filtered = []
            for item in results:
                item_types = set(item.get("test_type", "").split())
                if item_types.intersection(set(test_type_filter)):
                    filtered.append(item)
            results = filtered

        return results[:k]

    def _search_sbert(self, query: str, n: int) -> list[dict]:
        qvec = self._model.encode([query], normalize_embeddings=True)
        qvec = np.array(qvec, dtype=np.float32)
        scores, indices = self._index.search(qvec, min(n, len(self.catalog)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = self.catalog[idx].copy()
                item["_score"] = float(score)
                results.append(item)
        return results

    def _search_tfidf(self, query: str, n: int) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity
        qvec = self._tfidf.transform([query])
        sims = cosine_similarity(qvec, self._tfidf_matrix)[0]
        top_indices = sims.argsort()[::-1][:n]
        results = []
        for idx in top_indices:
            item = self.catalog[idx].copy()
            item["_score"] = float(sims[idx])
            results.append(item)
        return results

    def get_by_name(self, name: str) -> Optional[dict]:
        """Exact or fuzzy name lookup; returns first match or None."""
        name_lower = name.lower()
        for item in self.catalog:
            if name_lower in item.get("name", "").lower():
                return item
        return None

    def format_for_context(self, items: list[dict]) -> str:
        """
        Render retrieved items as a compact string for the LLM system prompt context.
        Keeps token usage low while preserving all decision-relevant fields.
        """
        lines = []
        for i, item in enumerate(items, 1):
            types = item.get("test_type", "?")
            remote = "remote" if item.get("remote_testing") else "in-person"
            adaptive = ", adaptive" if item.get("adaptive_irt") else ""
            dur = item.get("duration_minutes")
            dur_str = f", {dur} min" if dur else ""
            desc = item.get("description", "")[:200]
            lines.append(
                f"{i}. **{item['name']}** [type: {types}{dur_str}, {remote}{adaptive}]\n"
                f"   URL: {item['url']}\n"
                f"   {desc}"
            )
        return "\n\n".join(lines)
