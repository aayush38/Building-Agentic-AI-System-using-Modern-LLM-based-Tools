"""CLI demo: python run_cli.py data/churn.csv "Predict churn and recommend a model" """
import sys

sys.stdout.reconfigure(encoding="utf-8")

from agent.graph import run_agent

ICON = {"goal": "GOAL", "thought": "THOUGHT", "action": "ACTION", "observation": "OBSERVATION",
        "final_answer": "FINAL ANSWER"}

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/churn.csv"
    goal = sys.argv[2] if len(sys.argv) > 2 else "Recommend a baseline model for this dataset."
    name = path.split("/")[-1].split(".")[0]
    calls = tokens = 0
    try:
        for rec in run_agent(f"Dataset: {path}\nGoal: {goal}", run_name=name):
            if rec.get("tokens"):
                calls += 1
                tokens += rec["tokens"]
            extra = f" {rec['args']}" if "args" in rec else ""
            model = f" (model: {rec['model']})" if rec.get("model") else ""
            print(f"\n[{rec['step']}] {ICON.get(rec['kind'], rec['kind'])}{extra}{model}\n{rec['content']}")
        print(f"\n--- LLM calls: {calls}, total tokens: {tokens}")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
            sys.exit("Rate limit reached on every configured provider. Wait for the daily reset "
                     "or add another provider key to .env (see LLM_CHAIN).")
        raise
