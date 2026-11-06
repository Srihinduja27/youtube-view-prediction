import os
import sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation import validate_schema


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]
    if not os.path.exists(data_path):
        print(f"[ERROR] Target dataset not found at: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path, low_memory=False)
    passed = validate_schema(df, output_report_name="baseline_validation.csv")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
