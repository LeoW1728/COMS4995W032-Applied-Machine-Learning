import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class RetrievalResult:
    problem_id: str
    score: float
    question_snippet: str
    difficulty: Optional[str] = None
    url: Optional[str] = None
    starter_code: Optional[str] = None
    solution_snippet: Optional[str] = None


class ProblemDatabase:
    def __init__(self, artifact_dir: str = "data/reference_db"):
        self.artifact_dir = Path(artifact_dir)
        self.records: List[dict] = []
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.matrix: Optional[sparse.csr_matrix] = None

    def is_ready(self) -> bool:
        required = [
            self.artifact_dir / "apps.jsonl",
            self.artifact_dir / "tfidf_matrix.npz",
            self.artifact_dir / "tfidf_vectorizer_words.joblib",
            self.artifact_dir / "tfidf_vectorizer_chars.joblib",
        ]
        return all(p.exists() for p in required)

    def load(self):
        if not self.is_ready():
            raise FileNotFoundError(
                "Reference DB artifacts missing. Run scripts/build_reference_db.py and scripts/build_tfidf_index.py"
            )
        self.records = []
        with (self.artifact_dir / "apps.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
        self.word_vectorizer = joblib.load(self.artifact_dir / "tfidf_vectorizer_words.joblib")
        self.char_vectorizer = joblib.load(self.artifact_dir / "tfidf_vectorizer_chars.joblib")
        self.matrix = sparse.load_npz(self.artifact_dir / "tfidf_matrix.npz")
        logger.info("Loaded %d records into RAG DB", len(self.records))

    def _build_query_vec(self, query: str):
        word_vec = self.word_vectorizer.transform([query])
        char_vec = self.char_vectorizer.transform([query])
        return sparse.hstack([word_vec, char_vec]).tocsr()

    def search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        if self.matrix is None:
            raise RuntimeError("RAG DB not loaded. Call load() first.")
        query_vec = self._build_query_vec(query)
        sims = self.matrix.dot(query_vec.T).toarray().ravel()
        top_idx = np.argsort(-sims)[:k]
        results: List[RetrievalResult] = []
        for idx in top_idx:
            rec = self.records[int(idx)]
            score = float(sims[int(idx)])
            results.append(
                RetrievalResult(
                    problem_id=str(rec.get("problem_id")),
                    score=score,
                    question_snippet=_trim_text(rec.get("question", "")),
                    difficulty=rec.get("difficulty"),
                    url=rec.get("url"),
                    starter_code=_trim_text(rec.get("starter_code") or "", 400),
                    solution_snippet=_trim_text(_first_solution(rec.get("solutions")), 400),
                )
            )
        return results


def _trim_text(text: str, limit: int = 600) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _first_solution(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list) and raw:
        return str(raw[0])
    if isinstance(raw, str):
        return raw
    return str(raw)
