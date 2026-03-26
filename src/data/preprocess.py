"""Clean, filter, and deduplicate raw data."""

from __future__ import annotations

import logging

import sqlparse
from datasets import Dataset

logger = logging.getLogger(__name__)


def is_valid_sql(sql: str) -> bool:
    """Check if a string parses as valid SQL."""
    try:
        parsed = sqlparse.parse(sql)
        return len(parsed) > 0 and parsed[0].get_type() is not None
    except Exception:
        return False


def normalize_sql(sql: str) -> str:
    """Normalize SQL for deduplication: lowercase, strip, collapse whitespace."""
    return " ".join(sql.strip().lower().split())


def preprocess(ds: Dataset) -> Dataset:
    """Clean and deduplicate a Text-to-SQL dataset.

    Steps:
        1. Strip whitespace from all text fields.
        2. Drop rows with empty question, context, or answer.
        3. Validate that the answer is parseable SQL.
        4. Deduplicate on (normalized question, normalized answer).

    Returns:
        Cleaned dataset.
    """
    text_cols = ["question", "context", "answer"]
    original_len = len(ds)

    # 1. Strip whitespace
    ds = ds.map(
        lambda row: {k: row[k].strip() for k in text_cols},
        desc="Stripping whitespace",
    )

    # 2. Drop empty fields
    ds = ds.filter(
        lambda row: all(len(row[k]) > 0 for k in text_cols),
        desc="Removing empty rows",
    )
    logger.info("After empty-field filter: %d -> %d", original_len, len(ds))

    # 3. Validate SQL
    ds = ds.filter(
        lambda row: is_valid_sql(row["answer"]),
        desc="Validating SQL",
    )
    logger.info("After SQL validation: %d rows", len(ds))

    # 4. Deduplicate
    seen: set[tuple[str, str]] = set()
    keep_indices: list[int] = []

    for i, row in enumerate(ds):
        key = (normalize_sql(row["question"]), normalize_sql(row["answer"]))
        if key not in seen:
            seen.add(key)
            keep_indices.append(i)

    ds = ds.select(keep_indices)
    logger.info(
        "After deduplication: %d rows (removed %d dupes)",
        len(ds), original_len - len(ds),
    )

    return ds
