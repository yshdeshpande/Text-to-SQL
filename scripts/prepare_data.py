"""End-to-end data preparation: download, preprocess, format, split."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import concatenate_datasets

from src.data.download import download_all
from src.data.preprocess import preprocess
from src.data.analysis import compute_stats
from src.data.format import format_for_training  
from src.data.split import split_by_schema        

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")


def main() -> None:
    # 1. Download Spider + Gretel
    data = download_all()

    # 2. Merge Spider train + Gretel for training pool
    train_pool = concatenate_datasets([data["spider_train"], data["gretel"]])
    logger.info("Combined training pool: %d examples", len(train_pool))

    # 3. Clean & deduplicate training pool
    train_pool = preprocess(train_pool)

    # 4. Analyze
    stats = compute_stats(train_pool)
    logger.info("Training pool stats:\n%s", json.dumps(stats, indent=2))

    # 5. Format into instruction-tuning chat messages
    train_pool = format_for_training(train_pool)
    spider_dev = format_for_training(data["spider_dev"])

    # 6. Split training pool into train / val (by schema)
    #    Spider dev is kept separate as the final test set
    splits = split_by_schema(train_pool)
    splits["test"] = spider_dev

    # 7. Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_ds in splits.items():
        path = OUTPUT_DIR / split_name
        split_ds.save_to_disk(str(path))
        logger.info("Saved %s split: %d examples -> %s", split_name, len(split_ds), path)

    # Save stats
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
