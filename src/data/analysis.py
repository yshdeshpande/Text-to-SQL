"""Data distribution analysis and visualization."""

from __future__ import annotations

import logging
from collections import Counter

import sqlparse
from datasets import Dataset

logger = logging.getLogger(__name__)


def compute_stats(ds: Dataset) -> dict:
    """Compute summary statistics over a Text-to-SQL dataset.

    Returns a dict with:
        - total_examples: int
        - question_length: {min, max, mean, median}
        - context_length:  {min, max, mean, median}
        - answer_length:   {min, max, mean, median}
        - num_tables_per_example: {min, max, mean, median}
        - sql_keyword_counts: Counter of SQL keywords (SELECT, JOIN, WHERE, GROUP BY, etc.)
    """
    q_lens = [len(row["question"]) for row in ds]
    c_lens = [len(row["context"]) for row in ds]
    a_lens = [len(row["answer"]) for row in ds]

    num_tables = [row["context"].upper().count("CREATE TABLE") for row in ds]

    keyword_counter: Counter[str] = Counter()
    keywords = ["JOIN", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "UNION", "SUBQUERY", "LIKE", "IN", "BETWEEN", "DISTINCT", "LIMIT"]

    for row in ds:
        sql_upper = row["answer"].upper()
        for kw in keywords:
            if kw in sql_upper:
                keyword_counter[kw] += 1

    def _stats(vals: list[int]) -> dict:
        s = sorted(vals)
        n = len(s)
        return {
            "min": s[0],
            "max": s[-1],
            "mean": round(sum(s) / n, 1),
            "median": s[n // 2],
        }

    return {
        "total_examples": len(ds),
        "question_length": _stats(q_lens),
        "context_length": _stats(c_lens),
        "answer_length": _stats(a_lens),
        "num_tables_per_example": _stats(num_tables),
        "sql_keyword_counts": dict(keyword_counter.most_common()),
    }


def extract_table_names(context: str) -> list[str]:
    """Extract table names from CREATE TABLE statements."""
    tables = []
    for statement in sqlparse.parse(context):
        tokens = [t for t in statement.flatten() if not t.is_whitespace]
        for i, token in enumerate(tokens):
            if token.ttype is sqlparse.tokens.Keyword.DDL and token.normalized == "CREATE":
                # Look for the table name after "CREATE TABLE"
                for t in tokens[i + 1 :]:
                    if t.ttype is sqlparse.tokens.Name:
                        tables.append(t.value)
                        break
    return tables
