import argparse
import json
from pathlib import Path
from typing import Optional, List

try:
    # optional import; if datasets not installed, JSONL search still works
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # type: ignore


def _make_payload_from_row(row: dict) -> dict:
    return {
        "question": row.get("question", ""),
        "input_output": row.get("input_output", {}),
        "solutions": row.get("solutions", []),
        "starter_code": row.get("starter_code", ""),
        "meta": {
            "problem_id": row.get("problem_id"),
            "difficulty": row.get("difficulty"),
            "url": row.get("url"),
            "apps_config": row.get("apps_config", row.get("apps_config_name")),
            "apps_split": row.get("apps_split"),
        },
    }


def _search_jsonl_for_problem(path: Path, problem_id: str) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pid = rec.get("problem_id")
            if pid is None:
                continue
            if str(pid) == problem_id:
                return _make_payload_from_row(rec)
    return None


def _search_hf_apps(problem_id: str, configs: List[str], split: str, cache_dir: Optional[str]) -> Optional[dict]:
    if load_dataset is None:
        return None
    for cfg in configs:
        try:
            ds = load_dataset("codeparrot/apps", cfg, split=split, cache_dir=cache_dir)
        except Exception:
            # skip configs we can't load
            continue
        for row in ds:
            pid = row.get("problem_id")
            if pid is None:
                continue
            if str(pid) == problem_id:
                rec = dict(row)
                rec["apps_config"] = cfg
                rec["apps_split"] = split
                return _make_payload_from_row(rec)
    return None


def main():
    parser = argparse.ArgumentParser(description="Fetch an APPS problem by id for quick copy")
    parser.add_argument("--jsonl", default="data/reference_db/apps.jsonl", help="Local apps jsonl (from build_reference_db)")
    parser.add_argument("--problem_id", type=str, help="Problem id to fetch (e.g. 12345)")
    parser.add_argument("--configs", type=str, default="introductory,interview,competition",
                        help="Comma-separated APPS configs to search when not found locally")
    parser.add_argument("--split", type=str, default="test", help="HF split to search (default: test)")
    parser.add_argument("--cache_dir", type=str, default=None, help="HF datasets cache dir (optional)")
    parser.add_argument("--idx", type=int, default=None, help="Fallback: fetch by line index from local jsonl")

    args = parser.parse_args()

    if not args.problem_id and args.idx is None:
        parser.error("Either --problem_id or --idx must be provided")

    # Prefer local JSONL lookup first for speed/offline use
    jsonl_path = Path(args.jsonl)
    if args.problem_id:
        pid = str(args.problem_id)
        payload = _search_jsonl_for_problem(jsonl_path, pid)
        if payload:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        # Not found locally; try HF datasets using same logic as evaluate_batch
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        payload = _search_hf_apps(pid, configs, args.split, args.cache_dir)
        if payload:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        raise RuntimeError(f"Problem id {pid} not found in {jsonl_path} or in HF APPS (configs={configs}, split={args.split})")

    # Fallback: fetch by index in local JSONL (preserve original behavior)
    if args.idx is not None:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing {jsonl_path}. Run scripts/build_reference_db.py first.")
        with jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == args.idx:
                    record = json.loads(line)
                    payload = _make_payload_from_row(record)
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                    return
        raise IndexError(f"Index {args.idx} out of range")


if __name__ == "__main__":
    main()
