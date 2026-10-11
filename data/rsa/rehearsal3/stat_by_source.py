"""Rehearsal 3: each critique statistic's observed value per data source (no model; observed only).
argv: <model_loop dir>. Run from the staged harness repo."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.rsa.dataset import load_forced_choice
from src.rsa.loop.critique import critique_frame

ml = Path(sys.argv[1])
t = load_forced_choice(ml / "responses.csv")
df = critique_frame(t.frame, t.contexts, t.choices)
print("rows by source:", df["source"].value_counts().to_dict())
for r in (1, 2):
    for f in sorted((ml / f"round_{r}/critique/test_stats").glob("*.py")):
        ns = {"json": json, "np": np, "pd": pd}
        exec(f.read_text(), ns)
        fn = ns["test_statistic"]
        out = {"all": round(fn(df), 4)}
        for s, g in df.groupby("source"):
            # rows the statistic can see: rerun on that source alone
            out[s] = round(fn(g.reset_index(drop=True)), 4)
        lit = df[df["source"] != "auto_psych"].reset_index(drop=True)
        out["literature"] = round(fn(lit), 4)
        print(f"r{r} {f.stem}: {json.dumps(out)}")
