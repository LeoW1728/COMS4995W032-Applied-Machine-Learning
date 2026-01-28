import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("build_reference_db")


REQUIRED_FIELDS = ["question", "input_output", "solutions", "difficulty", "url", "starter_code", "problem_id"]


def _ensure_required_fields(sample: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_FIELDS if k not in sample]
    if missing:
        raise ValueError(f"APPS sample missing required fields: {missing}. Got keys={list(sample.keys())}")


def _sample_indices(n_total: int, n_pick: int, seed: int) -> List[int]:
    if n_pick > n_total:
        raise ValueError(f"Requested {n_pick} samples, but dataset only has {n_total}.")
    # Deterministic without loading all into memory
    import random
    rnd = random.Random(seed)
    return rnd.sample(range(n_total), n_pick)


def _collect_config(
    config: str,
    split: str,
    n_pick: int,
    seed: int,
    cache_dir: str | None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    LOGGER.info("Loading HF dataset: codeparrot/apps config=%s split=%s", config, split)
    ds = load_dataset(
        "codeparrot/apps",
        config,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    n_total = len(ds)
    idxs = _sample_indices(n_total, n_pick, seed=seed)

    records: List[Dict[str, Any]] = []
    ids: List[str] = []

    for i in idxs:
        row = dict(ds[i])
        _ensure_required_fields(row)

        # Keep only the fields we actually use at runtime.
        rec = {
            "problem_id": row.get("problem_id"),
            "difficulty": row.get("difficulty"),
            "url": row.get("url"),
            "question": row.get("question"),
            "input_output": row.get("input_output"),
            "starter_code": row.get("starter_code"),
            "solutions": row.get("solutions"),
            # provenance
            "apps_config": config,
            "apps_split": split,
        }
        records.append(rec)
        ids.append(str(rec["problem_id"]))

    return records, ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Build data/reference_db/apps.jsonl from Hugging Face APPS by difficulty config.")
    ap.add_argument("--out", type=str, default="data/reference_db/apps.jsonl", help="Output JSONL path")
    ap.add_argument("--split", type=str, default="train", help="HF split to sample from (train/test/...)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    ap.add_argument("--cache_dir", type=str, default=None, help="Optional HF cache directory")

    # Default counts per config
    ap.add_argument("--intro_n", type=int, default=100, help="Number of introductory problems to include")
    ap.add_argument("--interview_n", type=int, default=200, help="Number of interview problems to include")
    ap.add_argument("--competition_n", type=int, default=300, help="Number of competition problems to include")

    args = ap.parse_args()

    plan = [
        ("introductory", args.split, args.intro_n, args.seed + 101),
        ("interview", args.split, args.interview_n, args.seed + 202),
        ("competition", args.split, args.competition_n, args.seed + 303),
    ]

    all_records: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    for config, split, n_pick, seed in plan:
        if n_pick <= 0:
            continue
        recs, ids = _collect_config(config=config, split=split, n_pick=n_pick, seed=seed, cache_dir=args.cache_dir)
        all_records.extend(recs)
        all_ids.extend(ids)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write apps.jsonl
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write loaded ids for leakage-free evaluation sampling
    ids_path = out_path.parent / "loaded_ids.json"
    ids_path.write_text(json.dumps(sorted(set(all_ids)), ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Wrote %d records to %s", len(all_records), out_path)
    LOGGER.info("Wrote %d unique problem_ids to %s", len(set(all_ids)), ids_path)


if __name__ == "__main__":
    main()
