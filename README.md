# ML Model Advisor Agent

A ReAct-style agentic AI system. Give it a dataset and a goal; an LLM plans the analysis,
calls tools (profile, preprocess, train/evaluate, calculator), observes results and recommends
a baseline model with justification. No model training of the LLM: the focus is reasoning and
tool orchestration.

```
User goal -> LLM reasoning -> tool call -> observation -> next action -> ... -> final answer
```

## Stack
LangGraph ReAct agent, OpenRouter free LLM, pandas, scikit-learn, Streamlit.

## Setup
```
pip install -r requirements.txt
copy .env.example .env      # add OPENROUTER_API_KEY, pick a free tool-calling MODEL_NAME
python data/make_datasets.py
```

## Run
```
python run_cli.py data/churn.csv "Predict churn and recommend a model"
python -m streamlit run app.py
```
Every run writes a JSONL trace to `logs/`.

## Tools
| Tool | Purpose |
|---|---|
| `profile_dataset` | shape, dtypes, missing values |
| `preprocess_data` | target, drop columns, impute, encode, detect task and imbalance |
| `train_and_evaluate` | 5-fold CV for a chosen model and metric |
| `calculator` | safe arithmetic |
