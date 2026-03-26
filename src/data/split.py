"""Train/val/test splitting logic (split by database schema to avoid leakage)."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

from datasets import Dataset, concatenate_datasets

logger = logging.getLogger(__name__)

DEFAULT_VAL_RATIO = 0.1


def _schema_key(row: dict) -> str:
    """Return a grouping key for a row.

    Spider rows have a ``db_id`` which is the most reliable grouping key.
    Gretel rows only have ``context`` (CREATE TABLE DDL), so we hash it
    to get a stable, comparable key.
    """
    db_id = row.get("db_id")
    if db_id:
        return f"spider__{db_id}"
    # Hash the context so two identical schemas share a key
    return "gretel__" + hashlib.md5(row["context"].encode()).hexdigest()


def split_by_schema(
    ds: Dataset,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = 42,
) -> dict[str, Dataset]:
    """Split a dataset into train / val by schema group.

    All examples sharing the same schema end up in the same split,
    preventing data leakage where the model memorises schema-specific patterns.

    Args:
        ds: The full training pool (Spider train + Gretel, already preprocessed).
        val_ratio: Target fraction for the validation split.
        seed: Random seed for reproducibility (used to shuffle schema groups).

    Returns:
        ``{"train": Dataset, "val": Dataset}``
    """
    # 1. Group row indices by schema key
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(ds):
        groups[_schema_key(row)].append(idx)

    logger.info("Found %d unique schema groups across %d examples", len(groups), len(ds))

    # 2. Sort groups by key then shuffle deterministically
    import random
    rng = random.Random(seed)
    schema_keys = sorted(groups.keys())
    rng.shuffle(schema_keys)

    # 3. Greedily fill val until we hit the target ratio
    total = len(ds)
    target_val = int(total * val_ratio)

    val_indices: list[int] = []
    train_indices: list[int] = []
    val_count = 0

    for key in schema_keys:
        idxs = groups[key]
        if val_count < target_val:
            val_indices.extend(idxs)
            val_count += len(idxs)
        else:
            train_indices.extend(idxs)

    train_ds = ds.select(train_indices)
    val_ds = ds.select(val_indices)

    logger.info(
        "Split result — train: %d (%.1f%%), val: %d (%.1f%%)",
        len(train_ds), 100 * len(train_ds) / total,
        len(val_ds), 100 * len(val_ds) / total,
    )

    return {"train": train_ds, "val": val_ds}
