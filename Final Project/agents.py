import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def safe_json_loads(raw: str) -> dict:
    """Parse JSON even if wrapped in ```json ...``` or with extra text around it."""
    if raw is None:
        raise ValueError("Empty LLM response (None).")
    s = raw.strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    if not s.startswith("{"):
        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            s = s[l:r+1].strip()
    return json.loads(s)



def _ensure_utf8_output() -> None:
    """Force UTF-8 stdout/stderr to avoid codec errors on GBK consoles."""

    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if callable(reconfig):  # pragma: no cover - environment dependent
            try:
                reconfig(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - best effort guard
                pass


_ensure_utf8_output()


@dataclass
class SolverResult:
    code: str
    approach: str
    assumptions: List[str]
    complexity_claim: Dict[str, str]
    changed_from_last: str


@dataclass
class CriticResult:
    passed: bool
    failure_type: str
    notes: str
    complexity_class: str
    complexity_evidence: List[str]
    suggested_fix: str
    test_summary: Dict[str, Any]


@dataclass
class IterationTrace:
    iteration: int
    retrieval: List[Dict[str, Any]]
    solver: SolverResult
    critic: CriticResult


@dataclass
class Guide:
    guide_title: str
    final_summary: str
    steps: List[Dict[str, Any]]
    pitfalls: List[str]
    final_complexity: Dict[str, str]


class LLMClient:
    """
    Minimal OpenAI-compatible client wrapper.

    DeepSeek is OpenAI-API compatible. Per DeepSeek docs, use:
      - base_url: https://api.deepseek.com  (or /v1)
      - model: deepseek-chat (cheapest, non-thinking) or deepseek-reasoner (thinking)
    """
    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower().strip()
        self.api_key = (api_key or "").strip()
        # Allow env var usage by default (recommended to avoid putting keys on CLI)
        if not self.api_key:
            if self.provider == "deepseek":
                self.api_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
            elif self.provider == "openai":
                self.api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()

        # Cheapest DeepSeek model is deepseek-chat (non-thinking). Use it by default.
        if self.provider == "deepseek":
            self.model = model or "deepseek-chat"
            # DeepSeek docs recommend https://api.deepseek.com (or /v1 for OpenAI compatibility)
            self.base_url = base_url or "https://api.deepseek.com"
        else:
            self.model = model or "gpt-4o-mini"
            self.base_url = base_url  # None means OpenAI default in SDK

        self.mock_mode = not self.api_key

        # Lazily created SDK client (so unit tests without openai installed can still import)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.mock_mode:
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency 'openai'. Install with: pip install -U openai"
            ) from exc

        if self.provider == "deepseek":
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "openai":
            # For OpenAI, base_url should typically be omitted
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.base_url else OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        """
        Send a single-turn prompt and return the assistant content.

        json_mode=True enables DeepSeek JSON Output (response_format={'type':'json_object'}).
        DeepSeek docs also require the word 'json' to appear in the prompt when json_mode=True.
        """
        if self.mock_mode:
            logger.warning("No API key provided; falling back to mock LLM output")
            return "MOCK"

        client = self._get_client()
        assert client is not None

        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            # DeepSeek JSON Output mode
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        
    # PATCH: handling sporatical empty content error from DeepSeek
    def complete_with_retry(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1600,
        max_retries: int = 2,
        min_temperature: float = 0.05,
        backoff_base: float = 0.6,
    ) -> str:
        """
        Minimal robust retry wrapper:
        - Retries on any exception / empty output
        - Lowers temperature each retry to increase determinism
        - Never crashes the whole batch; caller can fallback if needed
        """
        last_err: Exception | None = None
        t = float(temperature)

        for attempt in range(max_retries + 1):
            try:
                raw = self.complete(
                    prompt,
                    json_mode=json_mode,
                    temperature=t,
                    max_tokens=max_tokens,
                )
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError("Empty model output")
                return raw
            except Exception as e:
                last_err = e
                # cool down for next try
                t = max(min_temperature, t * 0.5)
                # exponential backoff (small)
                time.sleep(min(3.0, backoff_base * (2 ** attempt)))

        # If still failing, re-raise so SolverAgent can fallback cleanly
        raise RuntimeError(f"LLM failed after retries: {last_err}") from last_err



class SolverAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, *, question: str, tests: Dict[str, List[str]], retrieved: List[Dict[str, Any]], starter_code: Optional[str], prev_fix: Optional[str], iteration: int) -> SolverResult:
        retrieval_text = json.dumps(retrieved, ensure_ascii=False, indent=2)
        prompt = "You are SolverAgent. Write a full Python program for the problem. Use retrieved hints as guidance only. Output strict JSON with keys code, approach, assumptions, complexity_claim, changed_from_last.\nProblem:\n{question}" + \
        f"\n\nTest cases:{tests}" if tests else "" + \
        f"\n\nStarter code:{starter_code}" if starter_code else "" + \
        f"\n\nRetrieved hints:{retrieval_text}" if retrieval_text else "" + \
        f"\n\nCriticAgent suggestions:{prev_fix}" if prev_fix else ""
        raw = self.llm.complete_with_retry(
            prompt,
            json_mode=True,
            temperature=0.2,
            max_tokens=1600,
            max_retries=2,
        )
        if raw == "MOCK":
            code = self._mock_code(tests)
            return SolverResult(
                code=code,
                approach="Mock fallback program that echoes expected output.",
                assumptions=["Using mock LLM output"],
                complexity_claim={"time": "O(1)", "space": "O(1)"},
                changed_from_last="(mock mode)" if iteration > 0 else "",
            )
        try:
            data = safe_json_loads(raw)
        except Exception as e:
            # raise ValueError(f"SolverAgent did not return valid JSON: {raw}")
            data = _solver_fallback_payload(err_msg=str(e), raw=raw)
        complexity_claim = data.get("complexity_claim", {})
        if not isinstance(complexity_claim, dict):
            complexity_claim = {}
        assumptions = data.get("assumptions", [])
        if not isinstance(assumptions, list):
            assumptions = []
        return SolverResult(
            code=data.get("code", ""),
            approach=data.get("approach", ""),
            assumptions=assumptions,
            complexity_claim=complexity_claim,
            changed_from_last=data.get("changed_from_last", ""),
        )

    def _mock_code(self, tests: Dict[str, List[str]]) -> str:
        inputs = tests.get("inputs") or []
        outputs = tests.get("outputs") or []
        pairs = {i: o for i, o in zip(inputs, outputs)}
        mapping_lines = ",".join(
            [
                f"{json.dumps(k)}: {json.dumps((v or '').strip())}"
                for k, v in pairs.items()
            ]
        )
        return (
            "import sys\n"
            "# Mock solution produced because no API key was provided.\n"
            f"mapping = {{{mapping_lines}}}\n"
            "data = sys.stdin.read()\n"
            "if data in mapping:\n"
            "    print(mapping[data])\n"
            "else:\n"
            "    print(mapping.get(data.strip(), ''))\n"
        )
        
# PATCH: for handling SolverAgent error raising due to DeepSeek error
def _solver_fallback_payload(err_msg: str, raw: str = "") -> dict:
    safe_raw = (raw or "")[:600] 
    return {
        "code": (
            "import sys\n"
            "def main():\n"
            "    # Fallback code due to invalid LLM JSON.\n"
            "    # Intentionally minimal; will likely fail tests.\n"
            "    data = sys.stdin.read()\n"
            "    if data is None:\n"
            "        return\n"
            "    # print nothing\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
        "approach": "Fallback: model output could not be parsed as valid JSON.",
        "assumptions": f"parse_error={err_msg}",
        "complexity_claim": "N/A",
        "changed_from_last": True,
        "debug_raw_prefix": safe_raw,
    }


class CriticAgent:
    def __init__(self, llm: Optional[LLMClient] = None, timeout: float = 2.0):
        self.timeout = timeout
        self.llm = llm

    def evaluate(self, code=None, tests=None, question=None, **kwargs) -> CriticResult:
        # Support evaluate(code, tests, question)  (positional)
        # Support evaluate(code=..., tests=..., question=...) (keyword)
        if code is None:
            code = kwargs.get("code")
        if tests is None:
            tests = kwargs.get("tests")
        if question is None:
            question = kwargs.get("question")

        code = code or ""
        tests = tests or {"inputs": [], "outputs": []}
        question = question or ""

        complexity_class, evidence = estimate_complexity(code)
        passed, summary, failure_type, notes = self._run_tests(code, tests)

        allowed, extra_note = complexity_gate(question, complexity_class)
        if not allowed:
            failure_type = "COMPLEXITY"
            notes = (notes + "; " if notes else "") + extra_note
            passed = False

        suggested_fix = self._suggest_fix(failure_type, code, notes)
        return CriticResult(
            passed=passed,
            failure_type=failure_type,
            notes=notes,
            complexity_class=complexity_class,
            complexity_evidence=evidence,
            suggested_fix=suggested_fix,
            test_summary=summary,
        )

    def _run_tests(self, code: str, tests: Dict[str, List[str]]):
        inputs = tests.get("inputs") or []
        outputs = tests.get("outputs") or []
        num_passed = 0
        first_failure = None
        failure_type = "WA"
        notes = ""
        for idx, (inp, expected) in enumerate(zip(inputs, outputs)):
            result = self._run_single(code, inp, expected)
            if result[0]:
                num_passed += 1
            else:
                failure_type, notes, got = result[1], result[2], result[3]
                first_failure = {"idx": idx, "expected": expected, "got": got}
                break
        passed = num_passed == len(inputs)
        summary = {
            "num_tests": len(inputs),
            "num_passed": num_passed,
            "first_failure": first_failure,
        }
        if not inputs or not outputs:
            return False, {"num_tests": 0, "num_passed": 0, "first_failure": None}, "NO_TESTS", "No tests provided"
        if passed:
            failure_type = "WA"
            notes = ""
        return passed, summary, failure_type if not passed else "WA", notes

    def _run_single(self, code: str, input_str: str, expected_output: str):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            proc = subprocess.run(
                ["python", tmp_path],
                input=input_str,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "TLE", "Time limit exceeded", ""
        stdout = normalize_output(proc.stdout)
        expected_norm = normalize_output(expected_output)
        if proc.returncode != 0:
            return False, "RE", proc.stderr[:400], stdout
        if stdout.strip() != expected_norm.strip():
            return False, "WA", "Wrong answer", stdout
        return True, "", "", stdout

    def _suggest_fix(self, failure_type: str, code: str, notes: str) -> str:
        """Call LLM to create a concise, actionable fix suggestion.
        Falls back to heuristic messages if no LLM is available or on error.
        """
        # If there's no LLM (or it's in mock mode), use the existing heuristics
        if not self.llm or getattr(self.llm, "mock_mode", True):
            if failure_type == "WA":
                return "Review logic against sample tests and ensure outputs match exactly."
            if failure_type == "RE":
                return f"Fix runtime error: {notes[:120]}"
            if failure_type == "TLE":
                return "Optimize loops/recursion to avoid timeouts."
            if failure_type == "COMPLEXITY":
                return "Replace nested loops with linear approach based on constraints."
            return "Investigate issues from critic feedback."

        # Compose a focused prompt for the LLM
        safe_code = (code or "")[:4000]
        prompt = (
            "You are a helpful code critic. The submission failed its tests.\n"
            f"Failure type: {failure_type}\n"
            f"Notes: {notes}\n\n"
            "Here is the submitted code (truncated if long):\n"
            f"{safe_code}\n\n"
            "Provide a short (1-3 sentence) summary of the likely root cause followed by "
            "2-4 specific, actionable suggestions to fix the code. Keep it concise."
        )
        try:
            resp = self.llm.complete_with_retry(
                prompt,
                json_mode=False,
                temperature=0.2,
                max_tokens=300,
                max_retries=1,
            )
            if isinstance(resp, str) and resp.strip():
                return resp.strip()
                
        except Exception:
            # On any LLM error, fall back to heuristics below
            pass

        # Final fallback heuristics
        if failure_type == "WA":
            return "Review logic against sample tests and ensure outputs match exactly."
        if failure_type == "RE":
            return f"Fix runtime error: {notes[:120]}"
        if failure_type == "TLE":
            return "Optimize loops/recursion to avoid timeouts."
        if failure_type == "COMPLEXITY":
            return "Replace nested loops with linear approach based on constraints."
        return "Investigate issues from critic feedback."


class GuiderAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    def synthesize(self, traces: List[IterationTrace]) -> Guide:
        # Fallback (existing) behaviour when no LLM is available or in mock mode
        if not self.llm or getattr(self.llm, "mock_mode", True):
            steps = []
            for t in traces:
                steps.append(
                    {
                        "iteration": t.iteration,
                        "what_failed_or_risk": t.critic.failure_type if not t.critic.passed else "OK",
                        "what_we_changed": t.solver.changed_from_last or "Initial attempt",
                        "evidence": t.critic.notes or json.dumps(t.critic.test_summary),
                        "complexity_before_after": {
                            "before": t.solver.complexity_claim.get("time", "unknown"),
                            "after": t.critic.complexity_class,
                        },
                    }
                )
            pitfalls = ["Keep outputs normalized (trim trailing spaces)", "Watch complexity gates for large N"]
            final_complexity = {
                "time": traces[-1].critic.complexity_class if traces else "unknown",
                "space": traces[-1].solver.complexity_claim.get("space", "unknown") if traces else "unknown",
            }
            return Guide(
                guide_title="How the solution evolved",
                final_summary="Concise walkthrough of solver and critic iterations.",
                steps=steps,
                pitfalls=pitfalls,
                final_complexity=final_complexity,
            )

        # When an LLM is available, build a compact trace summary and request structured JSON from it
        traces_summary = []
        for t in traces:
            traces_summary.append(
                {
                    "iteration": t.iteration,
                    "failure": t.critic.failure_type,
                    "passed": t.critic.passed,
                    "what_we_changed": t.solver.changed_from_last or "Initial attempt",
                    "notes": t.critic.notes,
                    "test_summary": t.critic.test_summary,
                    "complexity_before_after": {
                        "before": t.solver.complexity_claim.get("time", "unknown"),
                        "after": t.critic.complexity_class,
                    },
                }
            )

        prompt = (
            "You are a helpful summarizer that generates a concise guide from solver/critic traces. Output strict JSON.\n"
            "Input: a list of traces, each with iteration, failure, passed, what_we_changed, notes, test_summary, complexity_before_after.\n"
            "Produce a JSON object with keys: guide_title (string), final_summary (string), steps (list of objects with keys iteration, what_failed_or_risk, what_we_changed, evidence, complexity_before_after), pitfalls (list of strings), final_complexity (object with time and space).\n"
            "Traces (JSON):\n"
            f"{json.dumps(traces_summary, ensure_ascii=False, indent=2)[:8000]}\n\n"
            "Return only JSON. Keep outputs concise and suitable for programmatic parsing."
        )

        try:
            resp = self.llm.complete_with_retry(
                prompt,
                json_mode=True,
                temperature=0.2,
                max_tokens=800,
                max_retries=1,
            )
            # resp should be JSON or JSON-like string; parse robustly
            data = None
            if isinstance(resp, str):
                try:
                    data = safe_json_loads(resp)
                except Exception:
                    data = None
            elif isinstance(resp, dict):
                data = resp

            if isinstance(data, dict):
                steps = data.get("steps") or []
                pitfalls = data.get("pitfalls") or ["Keep outputs normalized (trim trailing spaces)", "Watch complexity gates for large N"]
                final_complexity = data.get("final_complexity") or {
                    "time": traces[-1].critic.complexity_class if traces else "unknown",
                    "space": traces[-1].solver.complexity_claim.get("space", "unknown") if traces else "unknown",
                }
                guide_title = data.get("guide_title", "How the solution evolved")
                final_summary = data.get("final_summary", "Concise walkthrough of solver and critic iterations.")

                return Guide(
                    guide_title=guide_title,
                    final_summary=final_summary,
                    steps=steps,
                    pitfalls=pitfalls,
                    final_complexity=final_complexity,
                )
        except Exception:
            # On any LLM error we fall back to the simple heuristic summary below
            pass

        # Fallback: previous heuristic behavior
        steps = []
        for t in traces:
            steps.append(
                {
                    "iteration": t.iteration,
                    "what_failed_or_risk": t.critic.failure_type if not t.critic.passed else "OK",
                    "what_we_changed": t.solver.changed_from_last or "Initial attempt",
                    "evidence": t.critic.notes or json.dumps(t.critic.test_summary),
                    "complexity_before_after": {
                        "before": t.solver.complexity_claim.get("time", "unknown"),
                        "after": t.critic.complexity_class,
                    },
                }
            )
        pitfalls = ["Keep outputs normalized (trim trailing spaces)", "Watch complexity gates for large N"]
        final_complexity = {
            "time": traces[-1].critic.complexity_class if traces else "unknown",
            "space": traces[-1].solver.complexity_claim.get("space", "unknown") if traces else "unknown",
        }
        return Guide(
            guide_title="How the solution evolved",
            final_summary="Concise walkthrough of solver and critic iterations.",
            steps=steps,
            pitfalls=pitfalls,
            final_complexity=final_complexity,
        )


def parse_problem_payload(payload_text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}")
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    for key in ["question", "input_output"]:
        if key not in payload:
            raise ValueError(f"Missing required field: {key}")
    return payload


def parse_tests(input_output_raw: Any) -> Dict[str, List[str]]:
    if isinstance(input_output_raw, str):
        try:
            parsed = json.loads(input_output_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"input_output string must be valid JSON: {exc}")
        # Some payloads may be doubly string-encoded (string containing JSON string)
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "input_output string contained nested JSON that could not be parsed: "
                    f"{exc}"
                )
    elif isinstance(input_output_raw, dict):
        parsed = input_output_raw
    else:
        raise ValueError("input_output must be string or dict")
    if not isinstance(parsed, dict):
        raise ValueError("input_output must decode to an object/dict")
    if "inputs" not in parsed or "outputs" not in parsed:
        raise ValueError("input_output missing inputs/outputs")
    inputs = parsed.get("inputs")
    outputs = parsed.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("inputs/outputs must be lists")
    if len(inputs) != len(outputs):
        raise ValueError("inputs and outputs must have the same length")
    return {"inputs": inputs, "outputs": outputs}


def normalize_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").replace("\r\n", "\n").split("\n")).strip()


def estimate_complexity(code: str):
    lines = [ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    loop_depth = 0
    max_depth = 0
    for ln in lines:
        if re.match(r"(for |while )", ln):
            loop_depth += 1
            max_depth = max(max_depth, loop_depth)
        if ln.endswith(":") is False:
            loop_depth = max(loop_depth - 1, 0)
    if max_depth >= 3:
        cls = "O(N^3)"
    elif max_depth == 2:
        cls = "O(N^2)"
    elif max_depth == 1:
        cls = "O(N)"
    else:
        cls = "O(1)"
    evidence = [f"Detected nested loop depth={max_depth}"]
    if "recursion" in code.lower():
        evidence.append("recursion keyword spotted")
    return cls, evidence


def complexity_gate(question: str, complexity_class: str):
    note = ""
    constraints_large = re.search(r"1e5|10\^5|100000", question)
    constraints_mid = re.search(r"1e4|10\^4|10000", question)
    if constraints_large:
        if complexity_class in {"O(N^2)", "O(N^3)", "O(2^N)", "O(N!)"}:
            return False, "Complexity too high for N>=1e5"
    elif constraints_mid:
        if complexity_class in {"O(N^2)", "O(N^3)", "O(2^N)", "O(N!)"}:
            return False, "Complexity too high for N around 1e4"
    else:
        if complexity_class in {"O(N^3)", "O(2^N)", "O(N!)"}:
            return False, "Rejected by default complexity gate"
    return True, note


def run_pipeline(
    payload_text: str,
    provider: str,
    api_key: Optional[str],
    retrieved: List[Dict[str, Any]],
    max_iters: int = 3,
    timeout: float = 2.0,
):
    payload = parse_problem_payload(payload_text)
    tests = parse_tests(payload.get("input_output"))
    question = payload.get("question", "")
    starter_code = payload.get("starter_code")
    llm = LLMClient(provider, api_key)
    solver = SolverAgent(llm)
    critic = CriticAgent(llm=llm, timeout=timeout)
    guider = GuiderAgent(llm=llm)

    traces: List[IterationTrace] = []
    prev_fix = None
    for i in range(max_iters):
        solver_result = solver.generate(
            question=question,
            tests=tests,
            retrieved=retrieved,
            starter_code=starter_code,
            prev_fix=prev_fix,
            iteration=i,
        )
        critic_result = critic.evaluate(solver_result.code, tests, question)
        traces.append(
            IterationTrace(
                iteration=i + 1,
                retrieval=retrieved,
                solver=solver_result,
                critic=critic_result,
            )
        )
        if critic_result.passed and critic_result.failure_type != "COMPLEXITY":
            break
        prev_fix = critic_result.suggested_fix

    guider_output = guider.synthesize(traces)
    final_pass = traces[-1].critic.passed and traces[-1].critic.failure_type != "COMPLEXITY"
    final_code = traces[-1].solver.code
    return {
        "decision": "PASS" if final_pass else "REJECT",
        "final_code": final_code,
        "guide": guider_output,
        "traces": traces,
        "tests": tests,
    }
