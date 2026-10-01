import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.common import RANDOM_STATE, TEST_SIZE, engineer_features, load_raw, regression_metrics, split_xy


def run_evaluation():
    print("Starting Model Evaluation...")

    # 1. Load test data and the trained model
    X_test_final = np.load(ROOT / "data/processed/X_test_final.npy")
    y_test = np.load(ROOT / "data/processed/y_test.npy")
    model = joblib.load(ROOT / "models/random_forest_mlflow.pkl")

    # 2. Predict and score
    y_pred = model.predict(X_test_final)
    m = regression_metrics(y_test, y_pred)

    print("\n--- Model Evaluation Report ---")
    print(f"MAE  (views)  : {m['mae']:,.2f}")
    print(f"RMSE (views)  : {m['rmse']:,.2f}")
    print(f"R2   (views)  : {m['r2']:.4f}")
    print(f"R2   (log)    : {m['log_r2']:.4f}")
    print("-------------------------------\n")

    # 3. Error analysis export: rebuild the same split to recover the original rows
    df, _ = engineer_features(load_raw())
    X, y = split_xy(df)
    _, X_test_raw, _, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    errors_df = df.loc[X_test_raw.index, ["video_id", "title", "category_id", "views"]].copy()
    errors_df["Predicted_Views"] = np.maximum(np.expm1(y_pred), 0).round(0)
    errors_df["Log_Error"] = y_pred - y_test  # positive = over-prediction

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    errors_df.sort_values("Log_Error", ascending=False).head(200).to_csv(out / "largest_overpredictions.csv", index=False)
    errors_df.sort_values("Log_Error", ascending=True).head(200).to_csv(out / "largest_underpredictions.csv", index=False)

    print("Evaluation complete and error analysis files saved!")


if __name__ == "__main__":
    run_evaluation()
