from __future__ import annotations

from src.pipelines.inner_loop.core import Dataset, Trial


def subjective_randomness_dataset(rows: list[dict], label: str = "subjective_randomness") -> Dataset:
    trials = []
    for row in rows:
        response = row.get("response") or ("left" if int(row.get("chose_left", 0)) else "right")
        trials.append(
            Trial(
                stimulus={"sequence_a": row["sequence_a"], "sequence_b": row["sequence_b"]},
                response=response,
                metadata=row,
            )
        )
    return Dataset(trials, ["left", "right"], label)
