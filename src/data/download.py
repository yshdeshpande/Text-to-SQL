"""Download and cache Text-to-SQL datasets."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset, load_dataset, concatenate_datasets

logger = logging.getLogger(__name__)

# Gretel complexity levels worth keeping (skip basic single-table stuff)
GRETEL_COMPLEX_CATEGORIES = [
    "aggregation",
    "single join",
    "multiple joins",
    "subqueries",
    "window functions",
]


def download_spider(cache_dir: str | Path | None = None) -> dict[str, Dataset]:
    """Download the Spider dataset with official train/dev splits.

    Returns:
        Dict with 'train' and 'dev' splits.
    """
    logger.info("Downloading Spider dataset ...")
    ds = load_dataset("xlangai/spider", cache_dir=cache_dir)

    train_ds = ds["train"]
    dev_ds = ds["validation"]

    logger.info("Spider train: %d examples, dev: %d examples", len(train_ds), len(dev_ds))
    return {"train": train_ds, "dev": dev_ds}


def download_gretel(
    cache_dir: str | Path | None = None,
    categories: list[str] | None = None,
) -> Dataset:
    """Download and filter the Gretel synthetic Text-to-SQL dataset.

    Filters to complex queries only (JOINs, subqueries, aggregations, window functions).

    Returns:
        Filtered dataset.
    """
    if categories is None:
        categories = GRETEL_COMPLEX_CATEGORIES

    logger.info("Downloading Gretel synthetic_text_to_sql dataset ...")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train", cache_dir=cache_dir)
    logger.info("Gretel total: %d examples", len(ds))

    # Filter to complex categories
    categories_lower = [c.lower() for c in categories]
    ds = ds.filter(
        lambda row: row.get("sql_complexity", "").lower() in categories_lower,
        desc="Filtering Gretel by complexity",
    )
    logger.info("Gretel after complexity filter: %d examples", len(ds))

    return ds


def normalize_gretel_to_spider_format(gretel_ds: Dataset) -> Dataset:
    """Normalize Gretel columns to match Spider-style format.

    Gretel has: sql_prompt (schema), sql_context (CREATE TABLEs), sql (answer), sql_explanation, sql_complexity
    We normalize to: question, context, answer
    """

    def _normalize(row: dict) -> dict:
        return {
            "question": row["sql_prompt"],
            "context": row["sql_context"],
            "answer": row["sql"],
            "source": "gretel",
        }

    return gretel_ds.map(_normalize, remove_columns=gretel_ds.column_names, desc="Normalizing Gretel")


def normalize_spider_format(spider_ds: Dataset) -> Dataset:
    """Normalize Spider columns to common format.

    Spider has: question, query, db_id, and schema info.
    We need: question, context (CREATE TABLEs), answer, source.
    """

    def _normalize(row: dict) -> dict:
        # Spider provides the query in 'query' column
        # Schema context comes from 'create_table' or must be reconstructed
        # The dataset typically has question, query, db_id
        return {
            "question": row["question"],
            "context": row.get("create_table", ""),
            "answer": row["query"],
            "db_id": row["db_id"],
            "source": "spider",
        }

    return spider_ds.map(_normalize, remove_columns=spider_ds.column_names, desc="Normalizing Spider")


def download_all(cache_dir: str | Path | None = None) -> dict:
    """Download and normalize both datasets.

    Returns:
        Dict with:
            - 'spider_train': normalized Spider train split
            - 'spider_dev': normalized Spider dev split (for final evaluation only)
            - 'gretel': normalized filtered Gretel dataset
    """
    spider = download_spider(cache_dir=cache_dir)
    gretel_raw = download_gretel(cache_dir=cache_dir)

    spider_train = normalize_spider_format(spider["train"])
    spider_dev = normalize_spider_format(spider["dev"])
    gretel = normalize_gretel_to_spider_format(gretel_raw)

    logger.info(
        "Final counts — Spider train: %d, Spider dev: %d, Gretel filtered: %d",
        len(spider_train), len(spider_dev), len(gretel),
    )

    return {
        "spider_train": spider_train,
        "spider_dev": spider_dev,
        "gretel": gretel,
    }
