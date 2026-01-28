import argparse
import json
import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("build_tfidf_index")


def _first_solution(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list) and raw:
        return str(raw[0])
    if isinstance(raw, str):
        return raw
    return str(raw)


def load_records(jsonl_path: Path) -> List[dict]:
    records: List[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_docs(records: List[dict]) -> List[str]:
    docs: List[str] = []
    for r in records:
        q = r.get("question", "") or ""
        sc = r.get("starter_code", "") or ""
        sol = _first_solution(r.get("solutions"))
        # Keep the document compact but informative; TF-IDF benefits from both text and code tokens.
        doc = f"{q}\n\n[STARTER]\n{sc}\n\n[SOLUTION]\n{sol}"
        docs.append(doc)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build TF-IDF artifacts for RAG from data/reference_db/apps.jsonl")
    ap.add_argument("--db_dir", type=str, default="data/reference_db", help="Directory containing apps.jsonl")
    ap.add_argument("--max_features_words", type=int, default=50000)
    ap.add_argument("--max_features_chars", type=int, default=50000)
    args = ap.parse_args()

    db_dir = Path(args.db_dir)
    jsonl_path = db_dir / "apps.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing {jsonl_path}. Run scripts/build_reference_db.py first.")

    records = load_records(jsonl_path)
    if not records:
        raise RuntimeError(f"No records found in {jsonl_path}")

    docs = build_docs(records)

    LOGGER.info("Fitting TF-IDF vectorizers on %d docs", len(docs))
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=args.max_features_words,
        lowercase=True,
        strip_accents="unicode",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=args.max_features_chars,
        lowercase=True,
        strip_accents="unicode",
    )

    Xw = word_vectorizer.fit_transform(docs)
    Xc = char_vectorizer.fit_transform(docs)

    X = sparse.hstack([Xw, Xc]).tocsr()

    # Persist artifacts
    sparse.save_npz(db_dir / "tfidf_matrix.npz", X)
    joblib.dump(word_vectorizer, db_dir / "tfidf_vectorizer_words.joblib")
    joblib.dump(char_vectorizer, db_dir / "tfidf_vectorizer_chars.joblib")

    stats = {
        "num_docs": len(docs),
        "vocab_words": int(len(word_vectorizer.vocabulary_)),
        "vocab_chars": int(len(char_vectorizer.vocabulary_)),
    }
    (db_dir / "tfidf_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    LOGGER.info("Saved TF-IDF artifacts to %s", db_dir)
    LOGGER.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
