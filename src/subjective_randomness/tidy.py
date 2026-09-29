"""Write tidy (long-format) rows to CSV for plotting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_tidy_csv(
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    columns: Sequence[str],
) -> None:
    """Write tidy rows to a CSV with a fixed column order.

    Fails loudly if any row is missing a declared column.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            missing = [c for c in columns if c not in row]
            if missing:
                raise KeyError(f"Tidy row missing columns {missing}: {dict(row)}")
            writer.writerow({c: row[c] for c in columns})
