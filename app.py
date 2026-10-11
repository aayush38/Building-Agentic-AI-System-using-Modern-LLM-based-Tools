"""Streamlit demo: streams the agent's reasoning trace live."""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.graph import PROVIDERS, run_agent

st.set_page_config(page_title="ML Model Advisor Agent", layout="wide")
st.title("ML Model Advisor Agent")
st.caption("ReAct agent (LangGraph + LLM) that profiles data, trains models and recommends one.")

# --- Sidebar: choose LLM provider ---------------------------------------------------
with st.sidebar:
    st.header("LLM provider")
    choice = st.selectbox("Provider", ["Auto (fallback chain)"] + list(PROVIDERS))
    if choice == "Auto (fallback chain)":
        os.environ["LLM_CHAIN"] = "gemini,groq,openrouter"
        st.caption("Tries Gemini, then Groq, then OpenRouter if one fails or is rate limited.")
    else:
        os.environ["LLM_CHAIN"] = choice

# --- Dataset ------------------------------------------------------------------------
upload = st.file_uploader("Upload your own CSV (optional)", type="csv")
if upload is not None:
    Path("data").mkdir(exist_ok=True)
    Path("data", Path(upload.name).name).write_bytes(upload.getvalue())

datasets = sorted(p.as_posix() for p in Path("data").glob("*.csv"))
default = datasets.index(f"data/{Path(upload.name).name}") if upload is not None else 0
path = st.selectbox("Dataset", datasets, index=default)
goal = st.text_input("Goal", "Recommend a baseline model and justify it.")

if st.button("Run agent") and path:
    final, models_used, scores, metric_name = None, set(), {}, None
    llm_calls = total_tokens = 0
    try:
        with st.status("Agent working...", expanded=True) as status:
            for rec in run_agent(f"Dataset: {path}\nGoal: {goal}", run_name=Path(path).stem):
                kind = rec["kind"]
                if rec.get("model"):
                    models_used.add(rec["model"])
                if rec.get("tokens"):
                    llm_calls += 1
                    total_tokens += rec["tokens"]
                if kind == "action":
                    st.markdown(f"**Action** `{rec['content']}` args: `{rec['args']}`")
                elif kind == "observation":
                    st.markdown(f"**Observation** (`{rec['tool']}`)")
                    st.code(rec["content"][:1500], language="json")
                    if rec["tool"] in ("compare_models", "train_and_evaluate"):
                        try:
                            out = json.loads(rec["content"])
                            for r in out.get("ranked", [out]):  # compare_models or single model
                                metric_name, value = [(k, v) for k, v in r.items() if k != "model"][0]
                                scores[r["model"]] = value
                        except (ValueError, KeyError, IndexError):
                            pass  # tool returned an error; the agent will retry
                elif kind == "thought":
                    st.markdown(f"**Thought:** {rec['content']}")
                elif kind == "final_answer":
                    final = rec["content"]
            status.update(label="Done", state="complete")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
            st.error("Rate limit reached on every configured provider. Wait for the daily "
                     "reset, or add another provider key to `.env` (see `LLM_CHAIN`).")
        else:
            st.error(f"Agent failed: {msg[:500]}")
        st.stop()

    if models_used:
        st.info("Answered by: " + ", ".join(sorted(models_used))
                + f"  |  {llm_calls} LLM calls, {total_tokens} tokens")
    if final:
        st.subheader("Final answer")
        st.markdown(final)
    if scores:
        st.subheader("Model comparison")
        st.caption(f"Mean 5-fold CV score ({metric_name}); higher is better.")
        st.bar_chart(pd.Series(scores, name=metric_name).sort_values(ascending=False))
