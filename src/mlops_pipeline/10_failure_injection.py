import json
import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation import validate_schema


def main():
    print("=" * 70)
    print("FAILURE INJECTION & FAULT TOLERANCE TESTING")
    print("=" * 70)

    raw_path = ROOT / "data/raw/youtube_combined.csv"
    if not raw_path.exists():
        raw_files = list((ROOT / "data/raw").glob("*videos.csv"))
        if raw_files:
            df = pd.read_csv(raw_files[0], low_memory=False).head(200).copy()
        else:
            print("[ERROR] No raw files available for failure injection.")
            sys.exit(1)
    else:
        df = pd.read_csv(raw_path, low_memory=False).head(200).copy()

    test_results = []

    # Test 1: Drop required column
    df_missing_col = df.drop(columns=["views"])
    print("[TEST 1] Missing critical column 'views':")
    passed_1 = validate_schema(df_missing_col, output_report_name="corrupted_simulation_errors.csv")
    test_results.append({
        "scenario": "Missing required column 'views'",
        "expected_pass": False,
        "actual_pass": passed_1,
        "caught": not passed_1,
    })

    # Test 2: Invalid negative views
    df_neg = df.copy()
    df_neg.loc[0:5, "views"] = -100
    print("\n[TEST 2] Negative views check:")
    passed_2 = validate_schema(df_neg, output_report_name="corrupted_simulation_errors.csv")
    test_results.append({
        "scenario": "Negative views (<0)",
        "expected_pass": False,
        "actual_pass": passed_2,
        "caught": not passed_2,
    })

    # Test 3: Malformed trending_date
    df_date = df.copy()
    df_date.loc[0:5, "trending_date"] = "INVALID_DATE_FORMAT"
    print("\n[TEST 3] Malformed trending_date format:")
    passed_3 = validate_schema(df_date, output_report_name="corrupted_simulation_errors.csv")
    test_results.append({
        "scenario": "Malformed trending_date regex mismatch",
        "expected_pass": False,
        "actual_pass": passed_3,
        "caught": not passed_3,
    })

    all_caught = all(t["caught"] for t in test_results)
    report = {
        "status": "PASSED" if all_caught else "FAILED",
        "scenarios_tested": len(test_results),
        "all_caught": all_caught,
        "results": test_results,
    }

    os.makedirs(ROOT / "artifacts", exist_ok=True)
    report_path = ROOT / "artifacts/failure_simulation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n======================================================================")
    print(f"FAILURE INJECTION SUMMARY: {'PASSED (All faults detected)' if all_caught else 'FAILED'}")
    print(f"Report saved to {report_path}")
    print("======================================================================")
    sys.exit(0 if all_caught else 1)


if __name__ == "__main__":
    main()
