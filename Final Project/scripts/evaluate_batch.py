import argparse
import datetime as _dt
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure repo root is on sys.path (so we can import agents.py / rag_engine.py)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets import load_dataset  # type: ignore

from agents import LLMClient, SolverAgent, CriticAgent, parse_problem_payload, parse_tests, run_pipeline
from rag_engine import ProblemDatabase, RetrievalResult

def _to_jsonable(x: Any):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, set):
        return [_to_jsonable(v) for v in x]
    if is_dataclass(x):
        return _to_jsonable(asdict(x))
    if hasattr(x, "__dict__"):
        return _to_jsonable(vars(x))
    return str(x)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _timestamp_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_payload_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Keep the same shape that app.py expects.
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


def _select_eval_problems(
    seed: int,
    n_eval: int,
    configs: List[str],
    split: str,
    cache_dir: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Deterministically sample n_eval problems from Hugging Face APPS across given configs.
    """
    rng = random.Random(seed)
    chosen: List[Dict[str, Any]] = []
    chosen_ids: set[str] = set()

    # Shuffle configs deterministically to avoid bias
    configs_shuffled = configs[:]
    rng.shuffle(configs_shuffled)

    # Round-robin pull from each config until we have enough
    per_config_seed_base = seed * 1000 + 7
    while len(chosen) < n_eval:
        progress = False
        for ci, cfg in enumerate(configs_shuffled):
            if len(chosen) >= n_eval:
                break
            ds = load_dataset(
                "codeparrot/apps",
                cfg,
                split=split,
                cache_dir=cache_dir,
            )

            # Use HF shuffle for deterministic ordering within this config
            ds_shuf = ds.shuffle(seed=per_config_seed_base + ci)
            # Scan until we find one usable item (stop early for speed)
            for row in ds_shuf:
                pid = row.get("problem_id")
                url = row.get("url")
                key = (str(url).strip() if url else f"{split}:{pid}")

                # If key is empty or already chosen, skip
                if not key or key in chosen_ids:
                    continue

                rec = dict(row)
                rec["apps_config"] = cfg
                rec["apps_split"] = split
                chosen.append(rec)
                chosen_ids.add(key)
                progress = True
                break

        if not progress:
            raise RuntimeError(
                "Could not find enough evaluation problems excluding reference ids. "
                "Try a different --seed, different --eval_split, or reduce --eval_n."
            )

    return chosen


def _retrieve(db: Optional[ProblemDatabase], question: str, rag_k: int) -> List[RetrievalResult]:
    if db is None or rag_k <= 0:
        return []
    return db.search(question, k=rag_k)


def eval_baseline_single(
    payload: Dict[str, Any],
    provider: str,
    api_key: Optional[str],
    retrieved: List[RetrievalResult],
    timeout: float,
) -> Dict[str, Any]:
    llm = LLMClient(provider=provider, api_key=api_key)
    solver = SolverAgent(llm)
    critic = CriticAgent(timeout=timeout)

    # IMPORTANT: parse_tests expects payload["input_output"], not the full payload
    try:
        tests = parse_tests(payload.get("input_output"))
    except Exception as e:
        # Don't crash the whole batch; record and continue
        return {
            "mode": "baseline_single",
            "passed": False,
            "gate_allowed": False,
            "complexity_estimate": None,
            "failure_type": "payload_parse_error",
            "first_failure": None,
            "final_code": "",
            "notes": f"parse_tests failed: {e}",
        }

    res = solver.generate(
        question=payload.get("question", ""),
        tests=tests,
        starter_code=payload.get("starter_code", ""),
        retrieved=[r.__dict__ for r in retrieved],
        prev_fix=None,
        iteration=0,
    )
    cres = critic.evaluate(
        question=payload.get("question", ""),
        code=res.code,
        tests=tests,
    )
    return {
        "mode": "baseline_single",
        "passed": bool(cres.passed),
        "gate_allowed": bool(getattr(cres, "gate_allowed", True)),
        "complexity_estimate": getattr(cres, "complexity", None),
        "failure_type": (
            cres.diagnosis.get("failure_type")
            if isinstance(getattr(cres, "diagnosis", None), dict)
            else getattr(cres, "failure_type", None)
        ),
        "first_failure": getattr(cres, "first_failure", None),
        "final_code": res.code,
    }

def eval_baseline_repeated(
    payload: Dict[str, Any],
    provider: str,
    api_key: Optional[str],
    retrieved: List[RetrievalResult],
    timeout: float,
    repeat_n: int,
    seed: int,
) -> Dict[str, Any]:
    llm = LLMClient(provider=provider, api_key=api_key)
    solver = SolverAgent(llm)
    critic = CriticAgent(timeout=timeout)

    try:
        tests = parse_tests(payload.get("input_output"))
    except Exception as e:
        return {
            "mode": "baseline_repeated",
            "passed": False,
            "gate_allowed": False,
            "attempts_used": 0,
            "repeat_n": repeat_n,
            "failure_type_counts": {"payload_parse_error": 1},
            "first_failure": None,
            "final_code": "",
            "complexity_estimate": None,
            "notes": f"parse_tests failed: {e}",
        }


    failures: List[str] = []
    attempts_used = 0
    best_code = ""
    first_failure = None
    gate_allowed_best = False
    complexity_best = None

    for i in range(repeat_n):
        attempts_used += 1
        res = solver.generate(
            question=payload.get("question", ""),
            tests=tests,
            starter_code=payload.get("starter_code", ""),
            retrieved=[r.__dict__ for r in retrieved],
            prev_fix=None,  # critical: independent attempts (no feedback)
            iteration=i + 1,  # small prompt variation
        )
        cres = critic.evaluate(
            question=payload.get("question", ""),
            code=res.code,
            tests=tests,
        )
        ftype = None
        if isinstance(getattr(cres, "diagnosis", None), dict):
            ftype = cres.diagnosis.get("failure_type")
        else:
            ftype = getattr(cres, "failure_type", None)
        if not ftype:
            ftype = "unknown"
        if first_failure is None:
            first_failure = getattr(cres, "first_failure", None)

        if bool(cres.passed) and bool(getattr(cres, "gate_allowed", True)):
            best_code = res.code
            gate_allowed_best = True
            complexity_best = getattr(cres, "complexity", None)
            return {
                "mode": "baseline_repeated",
                "passed": True,
                "gate_allowed": True,
                "attempts_used": attempts_used,
                "repeat_n": repeat_n,
                "failure_type_counts": _count_list(failures),
                "first_failure": first_failure,
                "final_code": best_code,
                "complexity_estimate": complexity_best,
            }

        failures.append(ftype)

    # none passed
    return {
        "mode": "baseline_repeated",
        "passed": False,
        "gate_allowed": False,
        "attempts_used": attempts_used,
        "repeat_n": repeat_n,
        "failure_type_counts": _count_list(failures),
        "first_failure": first_failure,
        "final_code": best_code,
        "complexity_estimate": complexity_best,
    }


def eval_agent(
    payload: Dict[str, Any],
    provider: str,
    api_key: Optional[str],
    retrieved: List[RetrievalResult],
    max_iters: int,
    timeout: float,
) -> Dict[str, Any]:
    out = run_pipeline(
        payload_text=json.dumps(payload, ensure_ascii=False),
        provider=provider,
        api_key=api_key,
        retrieved=[r.__dict__ for r in retrieved],
        max_iters=max_iters,
        timeout=timeout,
    )

    decision_raw = out.get("decision")
    traces = out.get("traces") or []

    # derive iters_used if not present
    iters_used = out.get("iters_used")
    if iters_used is None and isinstance(traces, list):
        iters_used = len(traces)

    def _extract_failure_type_from_traces(ts):
        if not isinstance(ts, list) or not ts:
            return None
        last = ts[-1]
        critic = getattr(last, "critic", None)
        if critic is None and isinstance(last, dict):
            critic = last.get("critic")
        if critic is None:
            return None
        # critic may be a dataclass-like object or dict
        if isinstance(critic, dict):
            # prefer diagnosis.failure_type if present
            diag = critic.get("diagnosis")
            if isinstance(diag, dict) and diag.get("failure_type"):
                return diag.get("failure_type")
            return critic.get("failure_type")
        return getattr(critic, "failure_type", None)

    # default outputs
    passed = False
    gate_allowed = True
    failure_type = None

    # Case A: decision is the original string "PASS"/"REJECT"
    if isinstance(decision_raw, str):
        passed = decision_raw.strip().upper() == "PASS"
        gate_allowed = True
        if not passed:
            failure_type = _extract_failure_type_from_traces(traces) or "unknown"

    # Case B: if you later change run_pipeline to return a dict decision, we still support it
    elif isinstance(decision_raw, dict):
        passed = bool(decision_raw.get("passed", False))
        gate_allowed = bool(decision_raw.get("gate_allowed", True))
        diagnosis = decision_raw.get("diagnosis", None)
        if isinstance(diagnosis, dict):
            failure_type = diagnosis.get("failure_type")

    else:
        # unexpected format
        passed = False
        gate_allowed = True
        failure_type = "bad_decision_format"

    return {
        "mode": "agent",
        "passed": passed,
        "gate_allowed": gate_allowed,
        "iters_used": iters_used,
        "failure_type": failure_type,
        "final_code": out.get("final_code"),
        "guide": out.get("guide"),
        "traces": traces,
    }


def _count_list(xs: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def _summarize(records: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    n = len(records)
    success = 0
    failures: Dict[str, int] = {}
    gate_rejects = 0

    iters_or_attempts: List[int] = []

    for r in records:
        ok = bool(r.get("passed")) and bool(r.get("gate_allowed", True))
        if ok:
            success += 1
            if mode == "agent" and r.get("iters_used") is not None:
                iters_or_attempts.append(int(r["iters_used"]))
            if mode == "baseline_repeated" and r.get("attempts_used") is not None:
                iters_or_attempts.append(int(r["attempts_used"]))
        else:
            ft = r.get("failure_type")
            if ft is None and isinstance(r.get("failure_type_counts"), dict):
                # for repeated: use the most common failure type in attempts
                counts = r["failure_type_counts"]
                if counts:
                    ft = max(counts.items(), key=lambda kv: kv[1])[0]
            if not ft:
                ft = "unknown"
            failures[ft] = failures.get(ft, 0) + 1

        if r.get("passed") and not r.get("gate_allowed", True):
            gate_rejects += 1

    success_rate = success / n if n else 0.0
    avg_steps = sum(iters_or_attempts) / len(iters_or_attempts) if iters_or_attempts else None

    return {
        "mode": mode,
        "n": n,
        "success": success,
        "success_rate": success_rate,
        "gate_rejects": gate_rejects,
        "failure_type_counts": failures,
        "avg_success_steps": avg_steps,  # iters for agent, attempts for repeated
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results", help="Root directory to write results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rag_ks", type=str, default="0,3", help="Comma-separated rag_k values to run, e.g. 0,3,5")
    ap.add_argument("--modes", type=str, default="baseline_single,baseline_repeated,agent",
                    help="Comma-separated: baseline_single, baseline_repeated, agent")
    ap.add_argument("--repeat_n", type=int, default=5, help="N for baseline_repeated (Pass@N best-of-N)")
    ap.add_argument("--max_iters", type=int, default=3, help="Max iters for agent (Pass@K)")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--provider", type=str, default="DeepSeek")
    ap.add_argument("--api_key", type=str, default=None)
    ap.add_argument("--workers", type=int, default=12,
                help="ThreadPool workers for parallel per-problem eval (I/O bound LLM calls). Use 1 to disable.")


    # Evaluation sampling from HF APPS (leakage-free)
    ap.add_argument("--eval_n", type=int, default=10, help="Number of evaluation problems to sample from APPS each run")
    ap.add_argument("--eval_configs", type=str, default="introductory,interview,competition",
                    help="Comma-separated APPS configs to sample evaluation problems from")
    ap.add_argument("--eval_split", type=str, default="test", help="HF split to sample evaluation problems from (test recommended)")
    ap.add_argument("--cache_dir", type=str, default=None)

    # Reference DB directory for RAG + loaded_ids
    ap.add_argument("--ref_db_dir", type=str, default="data/reference_db")

    args = ap.parse_args()

    out_root = Path(args.out_dir)
    run_dir = out_root / _timestamp_tag()
    _ensure_dir(run_dir)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rag_ks = [int(x.strip()) for x in args.rag_ks.split(",") if x.strip() != ""]
    eval_configs = [c.strip() for c in args.eval_configs.split(",") if c.strip()]

    ref_db_dir = Path(args.ref_db_dir)

    # Sample evaluation problems from HF (no exclusion needed if you eval on a different split/dataset)
    eval_rows = _select_eval_problems(
        seed=args.seed,
        n_eval=args.eval_n,
        configs=eval_configs,
        split=args.eval_split,
        cache_dir=args.cache_dir,
    )

    # Persist eval ids for reproducibility
    eval_keys = []
    for r in eval_rows:
        url = r.get("url")
        pid = r.get("problem_id")
        split = r.get("apps_split", args.eval_split)
        key = (str(url).strip() if url else f"{split}:{pid}")
        eval_keys.append(key)

    (run_dir / "eval_problem_keys.json").write_text(json.dumps(eval_keys, indent=2), encoding="utf-8")

    # Lazy-load RAG db only if needed.
    db: Optional[ProblemDatabase] = None
    if any(k > 0 for k in rag_ks):
        db = ProblemDatabase(str(ref_db_dir))
        if not db.is_ready():
            raise RuntimeError(
                "RAG artifacts missing (tfidf_matrix.npz + vectorizers). "
                "Build them first via scripts/build_reference_db.py and scripts/build_tfidf_index.py, "
                "or run with --rag_ks 0."
            )
        db.load()

    # Run experiments
    for rag_k in rag_ks:
        for mode in modes:
            records: List[Dict[str, Any]] = []
            t0 = time.time()

            def _process_one(idx_row: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
                idx, row = idx_row
                payload = _make_payload_from_row(row)
                question = payload.get("question", "")

                retrieved = _retrieve(db, question, rag_k)
                rec: Dict[str, Any] = {
                    "eval_index": idx,
                    "problem_id": payload.get("meta", {}).get("problem_id"),
                    "difficulty": payload.get("meta", {}).get("difficulty"),
                    "apps_config": payload.get("meta", {}).get("apps_config"),
                    "apps_split": payload.get("meta", {}).get("apps_split"),
                    "rag_k": rag_k,
                }

                start = time.time()
                try:
                    if mode == "baseline_single":
                        res = eval_baseline_single(
                            payload=payload,
                            provider=args.provider,
                            api_key=args.api_key,
                            retrieved=retrieved,
                            timeout=args.timeout,
                        )
                        rec.update(res)
                    elif mode == "baseline_repeated":
                        res = eval_baseline_repeated(
                            payload=payload,
                            provider=args.provider,
                            api_key=args.api_key,
                            retrieved=retrieved,
                            timeout=args.timeout,
                            repeat_n=args.repeat_n,
                            seed=args.seed,
                        )
                        rec.update(res)
                    elif mode == "agent":
                        res = eval_agent(
                            payload=payload,
                            provider=args.provider,
                            api_key=args.api_key,
                            retrieved=retrieved,
                            max_iters=args.max_iters,
                            timeout=args.timeout,
                        )
                        rec.update(res)
                    else:
                        raise ValueError(f"Unknown mode: {mode}")
                except Exception as e:
                    rec.update({
                        "mode": mode,
                        "passed": False,
                        "gate_allowed": False,
                        "failure_type": "exception",
                        "exception": repr(e),
                    })

                rec["runtime_sec"] = time.time() - start
                return rec

            items = list(enumerate(eval_rows))
            workers = max(1, int(getattr(args, "workers", 1)))

            if workers == 1:
                iterator = items
                if tqdm is not None:
                    iterator = tqdm(iterator, total=len(items), desc=f"{mode}/rag{rag_k}", leave=False)
                for item in iterator:
                    records.append(_process_one(item))
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_process_one, item) for item in items]
                    iterator = as_completed(futures)
                    if tqdm is not None:
                        iterator = tqdm(iterator, total=len(futures), desc=f"{mode}/rag{rag_k}", leave=False)
                    for fut in iterator:
                        records.append(fut.result())

            records.sort(key=lambda r: int(r.get("eval_index", 0)))

            # Write JSONL + summary
            tag = f"{mode}_ragk{rag_k}"
            jsonl_out = run_dir / f"{tag}.jsonl"
            with jsonl_out.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(_to_jsonable(r), ensure_ascii=False) + "\n")

            summary = _summarize(records, mode)
            if mode == "baseline_single":
                summary["pass_at"] = 1
            elif mode == "baseline_repeated":
                summary["pass_at"] = args.repeat_n
            elif mode == "agent":
                summary["pass_at"] = args.max_iters
            summary["rag_k"] = rag_k
            summary["seed"] = args.seed
            summary["eval_n"] = len(records)
            summary["eval_split"] = args.eval_split
            summary["eval_configs"] = eval_configs
            summary["total_runtime_sec"] = time.time() - t0

            summary_out = run_dir / f"{tag}_summary.json"
            summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            tqdm.write(f"[OK] {tag}: success_rate={summary.get('success_rate'):.3f} -> {jsonl_out} | {summary_out}")

    tqdm.write(f"[DONE] Results written to: {run_dir}")


if __name__ == "__main__":
    main()
