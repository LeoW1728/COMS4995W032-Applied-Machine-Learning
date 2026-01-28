# APPS RAG + Multi-Agent Demo

This repository contains a Streamlit demo that showcases a TF-IDF RAG engine combined with a three-agent workflow (Solver / Critic / Guider) to tackle algorithmic problems from the APPS dataset.

## Features
- Local TF-IDF retrieval built from the Hugging Face `codeparrot/apps` dataset.
- SolverAgent drafts Python solutions from the pasted APPS problem payload.
- CriticAgent executes code against provided APPS-format tests with timeout + complexity gate.
- GuiderAgent summarizes iteration history into a concise guide.
- Streamlit UI with three tabs: run pipeline, inspect debate trace, and view RAG status/analytics.

## Installation
```bash
pip install -r requirements.txt
```

## Build reference artifacts (one-time)
Log schema validation and export a small reference set plus TF-IDF index.
```bash
python scripts/build_reference_db.py --split train --limit 500
python scripts/build_tfidf_index.py
```
If your environment cannot reach Hugging Face directly, use the bundled offline stub:
```bash
HTTPS_PROXY= HTTP_PROXY= python scripts/build_reference_db.py --split test --limit 50 --allow-offline
python scripts/build_tfidf_index.py
```

## Run the app
```bash
streamlit run app.py
```
Open the provided local URL in your browser.

## Getting a pasteable APPS payload
Use the helper to print a JSON payload from the local reference DB:
```bash
python scripts/fetch_problem.py --idx 0
```
Copy the printed JSON into the "Problem JSON" textarea in the app.
If you prefer manual copy, open `data/reference_db/apps.jsonl` and paste a line.

## User flow
1. Open the Streamlit app.
2. Paste a single APPS problem JSON object containing `question` and `input_output`.
3. Choose provider (DeepSeek/OpenAI) and set API key.
4. Click **Run Full Pipeline** to trigger retrieval + multi-agent loop.
5. Review final decision/code, guide, and the full debate trace.

## Troubleshooting
- **Dataset download/cache issues**: clear your Hugging Face cache or set `HF_HOME` to a writeable path before running build scripts.
- **JSON parsing**: `input_output` and `solutions` can be strings containing JSON or already-parsed objects. The app will raise a clear error if parsing fails.
- **Missing artifacts**: build scripts must be run before the app can search; the UI will show the exact commands.
- **API keys**: provide valid keys for OpenAI or DeepSeek. Without a key, the app falls back to a minimal mock generator (tests will still execute honestly).
