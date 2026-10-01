import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split

from src.common import RANDOM_STATE, TEST_SIZE, regression_metrics, split_xy

DATA = ROOT / "data" / "processed" / "youtube_processed.csv"
MODEL = ROOT / "models" / "random_forest_model.pkl"
OUT = ROOT / "outputs"


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, low_memory=False)
    X, y = split_xy(df)
    model = joblib.load(MODEL)

    # Recreate the training split (80/20, random_state=42).
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_pred = model.predict(X_test)

    m = regression_metrics(y_test, y_pred)
    report = (
        "Metrics on original views scale\n"
        f"  MAE : {m['mae']:,.2f}\n  RMSE: {m['rmse']:,.2f}\n  R2  : {m['r2']:.6f}\n\n"
        "Metrics on log1p(views) scale\n"
        f"  MAE : {m['log_mae']:.6f}\n  RMSE: {m['log_rmse']:.6f}\n  R2  : {m['log_r2']:.6f}\n"
    )
    (OUT / "regression_report.txt").write_text(report, encoding="utf-8")

    # Actual vs predicted (log scale, 5000-point sample).
    idx = np.random.RandomState(RANDOM_STATE).choice(len(y_test), size=min(5000, len(y_test)), replace=False)
    plt.figure(figsize=(6, 6))
    plt.scatter(np.asarray(y_test)[idx], np.asarray(y_pred)[idx], alpha=0.3, s=8)
    lims = [float(np.min(y_test)), float(np.max(y_test))]
    plt.plot(lims, lims, "--", color="red")
    plt.title("Actual vs Predicted log(1 + views)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(OUT / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # 3-fold CV on the full pipeline (R2 on log scale).
    scores = cross_val_score(model, X, y, cv=3, scoring="r2")
    (OUT / "cross_validation_results.txt").write_text(
        "Validation R2 Scores (log scale): " + str(scores.tolist()) + "\n" +
        "Average Validation R2: " + str(scores.mean()), encoding="utf-8"
    )

    names = [n.split("__", 1)[-1] for n in model.named_steps["preprocessor"].get_feature_names_out()]
    importance = pd.Series(model.named_steps["model"].feature_importances_, index=names)
    importance.sort_values(ascending=False).to_csv(OUT / "feature_importance.csv", header=["importance"])

    return m["r2"], scores.mean()


if __name__ == "__main__":
    r2, cv_r2 = evaluate()
    print(f"R2 (original scale): {r2:.6f}")
    print(f"Average CV R2 (log scale): {cv_r2:.6f}")
