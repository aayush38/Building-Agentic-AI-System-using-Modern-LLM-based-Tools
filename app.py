"""Streamlit demo: streams the agent's reasoning trace live, then supports follow-up chat."""
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.graph import PROVIDERS, run_agent

st.set_page_config(page_title="ML Model Advisor Agent", layout="wide")
st.title("ML Model Advisor Agent")
st.caption("ReAct agent (LangGraph + LLM) that profiles data, compares models, tunes the best one "
           "and explains it. Ask follow-up questions: the agent remembers the conversation.")

# --- Session state: one conversation = one thread_id + a list of finished turns -----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id, st.session_state.turns = uuid.uuid4().hex, []


def new_session():
    st.session_state.thread_id, st.session_state.turns = uuid.uuid4().hex, []


# --- Sidebar: choose LLM provider ---------------------------------------------------------
with st.sidebar:
    st.header("LLM provider")
    choice = st.selectbox("Provider", ["Auto (fallback chain)"] + list(PROVIDERS))
    if choice == "Auto (fallback chain)":
        os.environ["LLM_CHAIN"] = "gemini,groq,openrouter"
        st.caption("Tries Gemini, then Groq, then OpenRouter if one fails or is rate limited.")
    else:
        os.environ["LLM_CHAIN"] = choice
    st.button("New session", on_click=new_session)


def friendly_error(e: Exception) -> str:
    msg = str(e)
    if "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
        return ("Rate limit reached on every configured provider. Wait for the daily reset, "
                "or add another provider key to `.env` (see `LLM_CHAIN`).")
    return f"Agent failed: {msg[:500]}"


def show_step(rec: dict):
    kind = rec["kind"]
    if kind == "action":
        st.markdown(f"**Action** `{rec['content']}` args: `{rec['args']}`")
    elif kind == "observation":
        st.markdown(f"**Observation** (`{rec['tool']}`)")
        st.code(rec["content"][:1500], language="json")
    elif kind == "thought":
        st.markdown(f"**Thought:** {rec['content']}")


def collect(turn: dict, rec: dict):
    """Update a turn's summary (models, tokens, scores, features) from one trace record."""
    if rec.get("model"):
        turn["models"].add(rec["model"])
    if rec.get("tokens"):
        turn["calls"] += 1
        turn["tokens"] += rec["tokens"]
    if rec["kind"] == "final_answer":
        turn["final"] = rec["content"]
    if rec["kind"] != "observation":
        return
    try:
        out = json.loads(rec["content"])
        if rec["tool"] in ("compare_models", "train_and_evaluate"):
            for r in out.get("ranked", [out]):  # compare_models or single model
                turn["metric"], value = [(k, v) for k, v in r.items() if k != "model"][0]
                turn["scores"][r["model"]] = value
        elif rec["tool"] == "analyze_best_model" and "top_features" in out:
            turn["tuning"] = {k: out[k] for k in ("model", "baseline_cv3", "tuned_cv3", "best_params")}
            turn["features"] = out["top_features"]
    except (ValueError, KeyError, IndexError):
        pass  # tool returned an error; the agent will retry


def show_turn(turn: dict):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        with st.expander(f"Reasoning trace ({len(turn['steps'])} steps)"):
            for rec in turn["steps"]:
                show_step(rec)
        if turn["models"]:
            st.info("Answered by: " + ", ".join(sorted(turn["models"]))
                    + f"  |  {turn['calls']} LLM calls, {turn['tokens']} tokens")
        if turn["final"]:
            st.markdown(turn["final"])
        if turn["scores"]:
            st.subheader("Model comparison")
            st.caption(f"Mean 5-fold CV score ({turn['metric']}); higher is better.")
            st.bar_chart(pd.Series(turn["scores"], name=turn["metric"]).sort_values(ascending=False))
        if turn["tuning"]:
            t = turn["tuning"]
            st.subheader("Tuning and feature importance")
            st.caption(f"{t['model']}: baseline {t['baseline_cv3']} -> tuned {t['tuned_cv3']} "
                       f"(3-fold CV), params {t['best_params']}")
            st.bar_chart(pd.Series(turn["features"], name="importance").sort_values(ascending=False))


def run_turn(question: str, prompt: str, run_name: str):
    """Run the agent for one user message, stream steps live, then store the finished turn."""
    turn = {"question": question, "steps": [], "models": set(), "calls": 0, "tokens": 0,
            "scores": {}, "metric": None, "tuning": None, "features": {}, "final": None}
    try:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"), st.status("Agent working...", expanded=True) as status:
            for rec in run_agent(prompt, run_name=run_name, thread_id=st.session_state.thread_id):
                collect(turn, rec)
                if rec["kind"] != "goal":
                    turn["steps"].append(rec)
                show_step(rec)
            status.update(label="Done", state="complete")
    except Exception as e:  # noqa: BLE001
        st.error(friendly_error(e))
        st.stop()
    st.session_state.turns.append(turn)
    st.rerun()  # redraw everything from stored turns


# --- Dataset + first question -------------------------------------------------------------
upload = st.file_uploader("Upload your own CSV (optional)", type="csv")
if upload is not None:
    Path("data").mkdir(exist_ok=True)
    Path("data", Path(upload.name).name).write_bytes(upload.getvalue())

datasets = sorted(p.as_posix() for p in Path("data").glob("*.csv"))
default = datasets.index(f"data/{Path(upload.name).name}") if upload is not None else 0
path = st.selectbox("Dataset", datasets, index=default)
goal = st.text_input("Goal", "Recommend a baseline model and justify it.")

if st.button("Run agent") and path:
    new_session()  # a fresh conversation about this dataset
    run_turn(f"{goal}  \n*(dataset: `{path}`)*", f"Dataset: {path}\nGoal: {goal}", Path(path).stem)

# --- Conversation so far + follow-ups -----------------------------------------------------
for turn in st.session_state.turns:
    show_turn(turn)

if st.session_state.turns:
    follow_up = st.chat_input("Ask a follow-up (e.g. 'why not random forest?' or "
                              "'redo it without tenure_months')")
    if follow_up:
        run_turn(follow_up, follow_up, "chat")
