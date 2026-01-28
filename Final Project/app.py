import json
import logging
import re
from typing import Any, Dict, List

import streamlit as st

from agents import LLMClient, SolverAgent, CriticAgent, GuiderAgent, IterationTrace, parse_tests, run_pipeline, parse_problem_payload
from rag_engine import ProblemDatabase

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="APPS RAG Multi-Agent Demo", layout="wide")


def main():
    st.title("RAG + Multi-Agent APPS Solver")

    db = ProblemDatabase()
    db_ready = db.is_ready()
    if db_ready:
        db.load()
    else:
        st.warning(
            "Reference DB missing. Build it with:\n"
            "python scripts/build_reference_db.py --split test --limit 500\n"
            "python scripts/build_tfidf_index.py"
        )

    tabs = st.tabs(["Problem & Run", "War Room / Agent Debate", "RAG & Analytics"])

    with tabs[0]:
        st.subheader("Paste APPS Problem JSON")
        sample_text = st.session_state.get("last_payload", "")
        payload_text = st.text_area("Problem JSON", sample_text, height=220, key="payload_editor")

        deepseek_api_key = st.text_input("DeepSeek API Key", type="password", key="deepseek_api_key")
        openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
        
        fetch_api = {
            "DeepSeek": deepseek_api_key,
            "OpenAI": openai_api_key
        }
        
        solver_provider = st.selectbox("Solver Provider", ["DeepSeek", "OpenAI"], index=0, key="solver_provider")
        critic_provider = st.selectbox("Critic Provider", ["DeepSeek", "OpenAI"], index=0, key="critic_provider")
        guider_provider = st.selectbox("Guider Provider", ["DeepSeek", "OpenAI"], index=0, key="guider_provider")
        
        max_iters = st.number_input("Max iterations", min_value=1, max_value=6, value=3, key="max_iters")
        rag_k = st.number_input("RAG top-k (0 disables retrieval)", min_value=0, max_value=10, value=3, key="rag_k")
        timeout = st.number_input("Timeout per test (sec)", min_value=0.5, max_value=30.0, value=2.0, step=0.5, key="timeout")

        run_btn = st.button("Run Full Pipeline", type="primary", key="run_btn")

        if run_btn:
            try:
                payload = parse_problem_payload(payload_text)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Payload parsing error: {exc}")
                st.stop()

            st.session_state["last_payload"] = payload_text
            question = payload.get("question", "") or ""
            st.session_state["last_question"] = question
            
            retrieved = None
            retrieved_dicts: List[Dict[str, Any]] = []
            if int(rag_k) > 0:
                if not db_ready:
                    st.error("RAG DB not built (but RAG top-k > 0). Build artifacts first, or set RAG top-k=0.")
                    st.stop()
                retrieved = db.search(question, k=int(rag_k))
                retrieved_dicts = [r.__dict__ for r in retrieved]

            try:
                tests = parse_tests(payload.get("input_output"))
                starter_code = payload.get("starter_code")
                
                solver = SolverAgent(llm=LLMClient(solver_provider, fetch_api[solver_provider]))
                critic = CriticAgent(llm=LLMClient(critic_provider, fetch_api[critic_provider]),
                                     timeout=float(timeout))
                guider = GuiderAgent(llm=LLMClient(guider_provider, fetch_api[guider_provider]))

                traces: List[IterationTrace] = []
                prev_fix = None
                
                for i in range(int(max_iters)):
                    solver_result = solver.generate(
                        question=question,
                        tests=tests,
                        retrieved=retrieved_dicts if i == 0 else None,
                        starter_code=starter_code,
                        prev_fix=prev_fix,
                        iteration=i,
                    )
                    critic_result = critic.evaluate(solver_result.code, tests, question)
                    traces.append(
                        IterationTrace(
                            iteration=i + 1,
                            retrieval=retrieved_dicts if i == 0 else [],
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
                
                results = {
                    "decision": "PASS" if final_pass else "REJECT",
                    "final_code": final_code,
                    "guide": guider_output,
                    "traces": traces,
                    "tests": tests,
                }
            except Exception as exc:  # noqa: BLE001
                st.error(f"Pipeline failed: {exc}")
                st.stop()

            st.session_state["run_results"] = results
            st.session_state["retrieved"] = retrieved_dicts if retrieved_dicts else []
        if "run_results" in st.session_state:
            res = st.session_state["run_results"]
            st.success(f"Final Decision: {res['decision']}")
            st.code(res.get("final_code", ""), language="python")
            guide = res.get("guide")
            if guide:
                st.write("### Teacher-Style Guide")
                st.markdown(render_guide_markdown(guide))

    with tabs[1]:
        st.subheader("War Room")
        traces: List = []
        if "run_results" in st.session_state:
            traces = st.session_state["run_results"].get("traces", [])
        if not traces:
            st.info("Run the pipeline to see agent debate.")
        else:
            for t in traces:
                with st.expander(f"Iteration {t.iteration}"):
                    st.write("#### Retrieval (evidence chain)")
                    render_retrieval_cards(t.retrieval, st.session_state.get("last_question", ""))
                    with st.expander("Raw retrieval JSON"):
                        st.json(t.retrieval)
                    st.write("#### Solver code")
                    st.code(t.solver.code, language="python")
                    st.write("Approach:", t.solver.approach)
                    st.write("#### Critic feedback (summary)")
                    st.markdown(render_critic_summary(t.critic))
                    st.write("Raw Critic JSON")
                    st.json(t.critic.__dict__)
                    if t.solver.changed_from_last:
                        st.write("Changed from last:", t.solver.changed_from_last)

    with tabs[2]:
        st.subheader("RAG & Analytics")
        st.write("DB ready:", db_ready)
        if db_ready:
            st.write("Indexed records:", len(db.records))
        if "retrieved" in st.session_state:
            st.write("Last retrieval set (evidence chain):")
            render_retrieval_cards(st.session_state["retrieved"], st.session_state.get("last_question", ""))
            with st.expander("Raw retrieval JSON"):
                st.json(st.session_state["retrieved"])
        if "run_results" in st.session_state:
            traces = st.session_state["run_results"].get("traces", [])
            if traces:
                complexity_series = {t.iteration: complexity_to_numeric(t.critic.complexity_class) for t in traces}
                st.line_chart(complexity_series)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _overlap_terms(query: str, snippet: str, top_n: int = 6) -> List[str]:
    q = set(_tokenize(query))
    s = _tokenize(snippet)
    overlap = [t for t in s if t in q and len(t) >= 3]
    # de-duplicate while preserving order
    seen = set()
    out = []
    for t in overlap:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= top_n:
            break
    return out


def render_retrieval_cards(retrieved: List[Dict[str, Any]], query: str) -> None:
    if not retrieved:
        st.info("Retrieval disabled or empty (RAG top-k=0).")
        return

    for i, r in enumerate(retrieved, start=1):
        score = r.get("score", 0.0)
        pid = r.get("problem_id", "")
        diff = r.get("difficulty") or "unknown"
        snippet = r.get("question_snippet") or ""
        url = r.get("url") or ""

        header = f"**#{i}**  score={score:.4f}  id={pid}  difficulty={diff}"
        st.markdown(header)
        if url:
            st.markdown(f"Source: {url}")
        overlap = _overlap_terms(query, snippet)
        if overlap:
            st.caption("Query overlap terms: " + ", ".join(overlap))
        if snippet:
            st.write(snippet)

        # Optional evidence payloads
        with st.expander("Show starter/solution snippets"):
            sc = r.get("starter_code") or ""
            sol = r.get("solution_snippet") or ""
            if sc:
                st.code(sc, language="python")
            if sol:
                st.code(sol, language="python")


def complexity_to_numeric(cls: str) -> int:
    order = {
        "O(1)": 1,
        "O(logN)": 2,
        "O(N)": 3,
        "O(NlogN)": 4,
        "O(N^2)": 5,
        "O(N^3)": 6,
        "O(2^N)": 7,
        "O(N!)": 8,
    }
    return order.get(cls, 0)


def render_guide_markdown(guide: Any) -> str:
    g = guide.__dict__ if hasattr(guide, "__dict__") else guide
    title = g.get("guide_title", "Guide")
    summary = g.get("final_summary", "")
    steps = g.get("steps", []) or []
    pitfalls = g.get("pitfalls", []) or []
    final_complexity = g.get("final_complexity", {}) or {}

    parts = [f"#### {title}", summary or ""]
    if steps:
        parts.append("**Iteration Recap**")
        for step in steps:
            parts.append(
                "- Iteration {iteration}: issue/risk — {issue}. Change — {change}. Evidence — {evidence}. Complexity {before} → {after}.".format(
                    iteration=step.get("iteration", "?"),
                    issue=step.get("what_failed_or_risk", ""),
                    change=step.get("what_we_changed", ""),
                    evidence=step.get("evidence", ""),
                    before=step.get("complexity_before_after", {}).get("before", "unknown"),
                    after=step.get("complexity_before_after", {}).get("after", "unknown"),
                )
            )
    if pitfalls:
        parts.append("**Pitfalls & Reminders**")
        for p in pitfalls:
            parts.append(f"- {p}")
    if final_complexity:
        parts.append(
            "**Final Complexity**: time {time}, space {space}".format(
                time=final_complexity.get("time", "unknown"),
                space=final_complexity.get("space", "unknown"),
            )
        )
    return "\n\n".join(parts)


def render_critic_summary(critic: Any) -> str:
    c = critic.__dict__ if hasattr(critic, "__dict__") else critic
    status = "approved" if c.get("passed") else "rejected"
    failure = c.get("failure_type", "?")

    test_summary = c.get("test_summary", {}) or {}
    num_tests = test_summary.get("num_tests", 0)
    num_passed = test_summary.get("num_passed", 0)
    first_failure = test_summary.get("first_failure")

    diagnosis = c.get("diagnosis", {}) or {}
    root = diagnosis.get("root_cause", "") or ""
    steps = diagnosis.get("fix_steps", []) or []
    directive = diagnosis.get("next_directive", "") or ""

    gate_allowed = c.get("gate_allowed", True)
    gate_note = c.get("gate_note", "") or ""

    lines = [
        f"The critic {status} the solution (type: {failure}).",
        f"Tests passed: {num_passed}/{num_tests}.",
    ]

    if first_failure:
        lines.append(
            "First failing case #{idx}: expected `{exp}` but got `{got}`.".format(
                idx=first_failure.get("idx"),
                exp=str(first_failure.get("expected", "")),
                got=str(first_failure.get("got", "")),
            )
        )

    if root:
        lines.append(f"Root cause: {root}")

    if steps:
        lines.append("Fix steps:")
        for s in steps:
            lines.append(f"- {s}")

    if directive:
        lines.append(f"Next directive to solver: {directive}")

    if not gate_allowed and gate_note:
        lines.append(f"Complexity gate: rejected — {gate_note}")
    else:
        lines.append(
            "Complexity gate: {cls}. Evidence: {evidence}".format(
                cls=c.get("complexity_class", "unknown"),
                evidence="; ".join(c.get("complexity_evidence", []) or []),
            )
        )

    return "\n".join(lines)



if __name__ == "__main__":
    main()
