"""Convert examples to instruction-tuning chat format."""

from __future__ import annotations

from datasets import Dataset

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "generate the correct SQL query. Output only the SQL query, nothing else."
)


def _build_user_message(question: str, context: str) -> str:
    """Build the user turn from a question and schema context."""
    return f"### Schema:\n{context}\n\n### Question:\n{question}"


def _build_messages(row: dict) -> dict:
    """Convert a single row into a chat messages list."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(row["question"], row["context"])},
        {"role": "assistant", "content": row["answer"]},
    ]
    return {"messages": messages}


def format_for_training(ds: Dataset) -> Dataset:
    """Add a ``messages`` column in chat format expected by SFTTrainer.

    Each row gets a list of three messages (system / user / assistant).
    The original columns are preserved alongside the new ``messages`` column.
    """
    return ds.map(_build_messages, desc="Formatting into chat messages")
