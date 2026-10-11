"""Generate small demo datasets (no downloads needed)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris

OUT = Path(__file__).parent


def make_churn(n=1500, seed=0):
    """Imbalanced binary classification with missing values and categoricals."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(n),
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charge": rng.normal(70, 25, n).round(2),
        "contract": rng.choice(["monthly", "yearly", "two_year"], n, p=[0.55, 0.25, 0.2]),
        "support_calls": rng.poisson(1.5, n),
        "payment_method": rng.choice(["card", "bank", "cash"], n),
    })
    logit = (-1.0 - 0.04 * df.tenure_months + 0.02 * df.monthly_charge
             + 0.35 * df.support_calls + (df.contract == "monthly") * 1.0)
    p = 1 / (1 + np.exp(-(logit - 1.5)))
    df["churn"] = (rng.random(n) < p).astype(int)
    for col in ["monthly_charge", "support_calls"]:
        df.loc[rng.random(n) < 0.08, col] = np.nan
    return df


if __name__ == "__main__":
    iris = load_iris(as_frame=True).frame.rename(columns={"target": "species"})
    iris.to_csv(OUT / "iris.csv", index=False)
    bc = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "malignant_benign"})
    bc.to_csv(OUT / "breast_cancer.csv", index=False)
    make_churn().to_csv(OUT / "churn.csv", index=False)
    print("Wrote iris.csv, breast_cancer.csv, churn.csv")
