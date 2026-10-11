"""Evaluate the agent against a brute-force search.

For every dataset, brute force scores ALL models with the same 5-fold CV the agent's tools use.
Then (unless --brute-only) the agent runs on the dataset and we check whether the model it
picked is the true best, or within TOLERANCE of it.

    python eval/run_eval.py --brute-only          # no API calls
    python eval/run_eval.py --provider groq       # runs the agent too
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from agent.tools import CLASSIFIERS, REGRESSORS, STATE, _evaluate, preprocess_data, profile_dataset  # noqa: E402

TOLERANCE = 0.01  # a pick within 0.01 of the best score counts as "practically best"

# dataset -> (target column, columns to drop). Add your own CSVs here.
DATASETS = {
    "data/iris.csv": ("species", []),
    "data/breast_cancer.csv": ("malignant_benign", []),
    "data/churn.csv": ("churn", ["customer_id"]),
}


def brute_force(path: str, target: str, drop: list[str]) -> dict:
    """Score every model; return the ranking, metric and task."""
    profile_dataset.invoke({"path": path})
    info = json.loads(preprocess_data.invoke({"target": target, "drop_columns": drop}))
    if info["task"] == "classification":
        metric, registry = ("f1_macro" if info.get("imbalanced") else "accuracy"), CLASSIFIERS
    else:
        metric, registry = "r2", REGRESSORS
    scores = {}
    for name in registry:
        r = _evaluate(name, metric)
        if "model" in r:
            scores[name] = r[metric]
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {"metric": metric, "task": info["task"], "ranking": ranking, "scores": scores}


def run_agent_once(path: str) -> dict:
    """Run the agent and pull out its pick, calls, tokens, time and tool errors."""
    from agent.graph import run_agent

    start, calls, tokens, errors, pick, top_compared, models = time.time(), 0, 0, 0, None, None, set()
    for rec in run_agent(f"Dataset: {path}\nGoal: Recommend a baseline model and justify it.",
                         run_name="eval_" + Path(path).stem):
        if rec.get("tokens"):
            calls += 1
            tokens += rec["tokens"]
        if rec.get("model"):
            models.add(rec["model"])
        if rec["kind"] == "action" and rec["content"] == "analyze_best_model":
            pick = rec["args"].get("model_name")  # the model the agent chose to recommend
        if rec["kind"] == "observation":
            try:
                out = json.loads(rec["content"])
            except ValueError:
                continue
            errors += "error" in out or bool(out.get("errors"))
            if rec["tool"] == "compare_models" and out.get("ranked") and top_compared is None:
                top_compared = out["ranked"][0]["model"]
    return {"pick": pick or top_compared, "llm_calls": calls, "tokens": tokens,
            "seconds": round(time.time() - start, 1), "tool_errors": errors,
            "llm_models": sorted(models)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brute-only", action="store_true", help="skip the agent (zero API calls)")
    ap.add_argument("--provider", help="pin one provider, e.g. groq (sets LLM_CHAIN)")
    ap.add_argument("--datasets", nargs="*", help="subset of dataset paths")
    args = ap.parse_args()
    if args.provider:
        os.environ["LLM_CHAIN"] = args.provider

    rows = []
    for path, (target, drop) in DATASETS.items():
        if args.datasets and path not in args.datasets:
            continue
        bf = brute_force(path, target, drop)
        best_name, best_score = bf["ranking"][0]
        row = {"dataset": path, "task": bf["task"], "metric": bf["metric"],
               "best_model": best_name, "best_score": best_score,
               "ranking": [f"{n} {s}" for n, s in bf["ranking"]]}
        if not args.brute_only:
            try:
                ag = run_agent_once(path)
                gap = round(best_score - bf["scores"].get(ag["pick"], float("nan")), 4)
                row.update(ag, gap=gap, exact=ag["pick"] == best_name,
                           within_tol=gap <= TOLERANCE)
            except Exception as e:  # noqa: BLE001  (e.g. rate limit)
                row["agent_error"] = str(e)[:200]
        rows.append(row)
        print(f"{path}: best={best_name} ({best_score})"
              + (f" | agent={row.get('pick')} gap={row.get('gap')}" if "pick" in row else ""))

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"eval_{stamp}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = ["| Dataset | Metric | True best (all models) | Agent pick | Gap | Exact | Within 0.01 | LLM calls | Tokens | Sec |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "pick" in r:
            lines.append(f"| {r['dataset']} | {r['metric']} | {r['best_model']} ({r['best_score']}) | {r['pick']} | "
                         f"{r['gap']} | {'yes' if r['exact'] else 'no'} | {'yes' if r['within_tol'] else 'no'} | "
                         f"{r['llm_calls']} | {r['tokens']} | {r['seconds']} |")
        else:
            note = r.get("agent_error", "agent skipped")
            lines.append(f"| {r['dataset']} | {r['metric']} | {r['best_model']} ({r['best_score']}) | - | - | - | - | - | - | {note[:40]} |")
    done = [r for r in rows if "pick" in r]
    if done:
        n = len(done)
        lines += ["", f"Agent picked the true best model in {sum(r['exact'] for r in done)}/{n} datasets "
                      f"and within {TOLERANCE} of it in {sum(r['within_tol'] for r in done)}/{n}. "
                      f"Mean {sum(r['llm_calls'] for r in done)/n:.1f} LLM calls and "
                      f"{sum(r['tokens'] for r in done)/n:.0f} tokens per run."]
    md = "\n".join(lines)
    (out_dir / f"eval_{stamp}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
