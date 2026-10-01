import sys
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    logging.info("Pipeline started")
    preprocess()
    train()
    r2, cv_r2 = evaluate()
    logging.info("R2 (views scale): %.6f", r2)
    logging.info("CV R2 (log scale): %.6f", cv_r2)
    logging.info("Pipeline completed")
    print("Baseline pipeline completed successfully.")
