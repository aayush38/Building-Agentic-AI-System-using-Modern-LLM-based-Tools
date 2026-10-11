# ML Model Advisor Agent

An agentic AI system: give it a dataset and a goal, and an LLM plans the analysis, calls tools,
observes the results and recommends a baseline ML model with evidence. The LLM is not trained;
the focus is reasoning, planning and tool orchestration (ReAct).

```
User goal -> LLM reasoning -> tool call -> observation -> next action -> ... -> final answer
```

## What the agent does
1. `profile_dataset` – shape, dtypes, missing values, unique counts
2. `preprocess_data` – choose target, drop ID columns, impute, one-hot encode, detect task and class imbalance
3. `compare_models` – 5-fold CV for 4-5 models in one call (linear, trees, boosting, KNN/SVM/naive Bayes)
4. `analyze_best_model` – small grid-search tuning of the winner plus permutation feature importance
5. Final answer – score table, recommendation with reason, top features, caveat

`train_and_evaluate` (single model) and `calculator` are available for retries and arithmetic.
If a tool returns an error (for example an unknown model name), the agent reads it and recovers.

## Features
- **ReAct agent** built with LangGraph; every Thought / Action / Observation is logged to `logs/*.jsonl`
- **Multi-provider LLM fallback chain**: Gemini, Groq, OpenRouter (and xAI); if one is rate limited the next is tried (`LLM_CHAIN` in `.env`)
- **Streamlit app**: live reasoning trace, provider picker, CSV upload, model-comparison and feature-importance charts, token/call counter
- **Follow-up chat with memory** (LangGraph checkpointer): ask "why not random forest?" or "redo it without tenure_months"
- **Token-efficient**: about 5 LLM calls and 6k tokens per run (batched model comparison, compact observations, step cap)
- **Evaluation harness**: compares the agent's pick against a brute-force search over all models

## Setup
```
pip install -r requirements.txt
copy .env.example .env      # then add at least one API key
python data/make_datasets.py
```
`.env` keys: `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY` (any one is enough), optional `GEMINI_MODEL`, `GROQ_MODEL`, `MODEL_NAME`, `LLM_CHAIN` (for example `gemini,groq,openrouter`).
Free-tier model names change, so check each provider's model list if you get "model not found".

## Run
```
python run_cli.py data/churn.csv "Predict churn and recommend a baseline model"
python -m streamlit run app.py
```

## Evaluate
```
python eval/run_eval.py --brute-only        # no API calls: best model per dataset
python eval/run_eval.py --provider groq     # also runs the agent and compares its pick
```
Results are written to `eval/results/` as markdown and JSON.

## Tools
| Tool | Purpose |
|---|---|
| `profile_dataset` | shape, dtypes, missing values |
| `preprocess_data` | target, drop columns, impute, encode, detect task and imbalance |
| `compare_models` | batched 5-fold CV for several models, ranked |
| `analyze_best_model` | grid-search tuning + permutation feature importance |
| `train_and_evaluate` | single-model CV (retry or extend) |
| `calculator` | safe arithmetic |

## Project layout
`agent/` (graph, tools, prompts, logger) · `data/` (sample CSVs) · `eval/` (harness) · `logs/` (traces) · `app.py` · `run_cli.py`
