---
name: critique-stat-no-imports
description: PPC test_stat files must not contain any import (even pandas) — harness rejects them
metadata:
  type: feedback
---
In critique rounds, test_stats/*.py with `import pandas` were all rejected ("forbidden import pandas"), so iter-4's first attempt produced zero statistics. np/pd/math are pre-injected; write no import lines at all.

**Why:** the whole critique round is lost if every file fails the import check.
**How to apply:** write the files first, use only np/pd/math, smoke-test by exec'ing with those globals. Parse sets with `str.split(",", expand=True).apply(pd.to_numeric)`. See [[iter4-critique-proxy-stats]].
