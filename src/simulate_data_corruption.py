from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/youtube_combined.csv"

if not RAW.exists():
    from src.common import load_raw
    df = load_raw().head(100)
else:
    df = pd.read_csv(RAW, low_memory=False).head(100)

corrupted = df.copy()
corrupted.loc[0:10, "views"] = -500
corrupted.to_csv(ROOT / "data/raw/corrupted_sample.csv", index=False)
print("Saved corrupted sample to data/raw/corrupted_sample.csv")
