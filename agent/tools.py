"""Tools the ML Advisor agent can call. Each returns a JSON string (the Observation)."""
import ast
import json
import operator as op
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent

# Shared working state between tool calls (the agent's "workspace").
STATE = {"df": None, "X": None, "y": None, "task": None, "target": None}

CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=0),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
    "knn": lambda: KNeighborsClassifier(),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=0),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=100, random_state=0),
    "svm": lambda: SVC(),
    "naive_bayes": lambda: GaussianNB(),
}
REGRESSORS = {
    "linear_regression": lambda: LinearRegression(),
    "decision_tree": lambda: DecisionTreeRegressor(random_state=0),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=0),
    "knn": lambda: KNeighborsRegressor(),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=0),
    "extra_trees": lambda: ExtraTreesRegressor(n_estimators=100, random_state=0),
    "svm": lambda: SVR(),
    "ridge": lambda: Ridge(),
    "lasso": lambda: Lasso(),
}
SCALED = {"knn", "logistic_regression", "svm", "ridge", "lasso"}  # scale-sensitive models


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


@tool
def profile_dataset(path: str) -> str:
    """Load a CSV and profile it (rows; per column: dtype, missing count, unique count). Call first."""
    try:
        df = pd.read_csv(_resolve(path))
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"Could not read {path}: {e}"})
    STATE.update(df=df, X=None, y=None, task=None, target=None)
    # Compact "dtype,missing,unique" strings keep the observation (re-sent every turn) small.
    cols = {c: f"{df[c].dtype},{int(df[c].isna().sum())},{int(df[c].nunique())}"
            for c in df.columns}
    return json.dumps({"rows": len(df), "columns(dtype,missing,unique)": cols})


@tool
def preprocess_data(target: str, drop_columns: list[str] | None = None) -> str:
    """Set the target, drop columns (IDs), impute, one-hot encode; reports task type and class balance."""
    df = STATE["df"]
    if df is None:
        return json.dumps({"error": "No dataset loaded. Call profile_dataset first."})
    if target not in df.columns:
        return json.dumps({"error": f"Target '{target}' not in columns {list(df.columns)}"})
    drop = [c for c in (drop_columns or []) if c in df.columns and c != target]
    data = df.drop(columns=drop).dropna(subset=[target])
    y = data[target]
    X = data.drop(columns=[target])
    for c in X.columns:
        if X[c].dtype.kind in "if":
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna(X[c].mode().iloc[0])
    X = pd.get_dummies(X, drop_first=True).astype(float)
    is_class = y.dtype.kind not in "f" or y.nunique() <= 10
    out = {"task": "classification" if is_class else "regression",
           "n_features": X.shape[1], "dropped": drop}
    if is_class:
        counts = y.value_counts(normalize=True).round(3).to_dict()
        out["class_balance"] = {str(k): v for k, v in counts.items()}
        out["imbalanced"] = bool(max(counts.values()) > 0.7)
    STATE.update(X=X, y=y, task=out["task"], target=target)
    return json.dumps(out)


def _evaluate(model_name: str, metric: str) -> dict:
    """5-fold CV for one model on the preprocessed data. Returns a result or an error dict."""
    if STATE["X"] is None:
        return {"error": "Data not preprocessed. Call preprocess_data first."}
    task = STATE["task"]
    registry = CLASSIFIERS if task == "classification" else REGRESSORS
    if model_name not in registry:
        return {"error": f"Unknown model '{model_name}' for {task}. Choose from {list(registry)}"}
    if metric == "auto":
        metric = "accuracy" if task == "classification" else "r2"
    try:
        model = registry[model_name]()
        X = STATE["X"]
        if model_name in SCALED:
            X = (X - X.mean()) / X.std().replace(0, 1)
        res = cross_validate(model, X, STATE["y"], cv=5, scoring=[metric])
    except Exception as e:  # noqa: BLE001
        return {"error": f"Training failed for {model_name}: {e}"}
    return {"model": model_name, metric: round(float(np.mean(res[f"test_{metric}"])), 4)}


MODEL_HELP = ("classification: logistic_regression, decision_tree, random_forest, extra_trees, "
              "gradient_boosting, knn, svm, naive_bayes; regression: linear_regression, ridge, lasso, "
              "decision_tree, random_forest, extra_trees, gradient_boosting, knn, svm. "
              "metric: accuracy, f1_macro (imbalanced), r2, or auto.")


@tool(description="Train SEVERAL models with 5-fold CV in one call and return scores, best "
                  "first. Prefer this over repeated train_and_evaluate calls. " + MODEL_HELP)
def compare_models(model_names: list[str], metric: str = "auto") -> str:
    results = [_evaluate(m, metric) for m in model_names]
    ok = sorted((r for r in results if "model" in r),
                key=lambda r: list(r.values())[1], reverse=True)
    errors = [r["error"] for r in results if "error" in r]
    return json.dumps({"ranked": ok, **({"errors": errors} if errors else {})})


@tool(description="Train ONE model with 5-fold CV (to retry or add a single model). " + MODEL_HELP)
def train_and_evaluate(model_name: str, metric: str = "auto") -> str:
    return json.dumps(_evaluate(model_name, metric))


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '(0.91 - 0.87) / 0.87 * 100'."""
    try:
        return json.dumps({"result": _eval(ast.parse(expression, mode="eval").body)})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"Cannot evaluate: {e}"})


TOOLS = [profile_dataset, preprocess_data, compare_models, train_and_evaluate, calculator]
