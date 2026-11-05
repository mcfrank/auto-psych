---
name: critique-stat-no-imports
description: PPC test-stat files must not contain `import pandas`/`import numpy` — the harness rejects them; np/pd/math are pre-injected
metadata:
  type: feedback
---
In critique rounds, a test_stats file containing `import pandas` is set aside as "forbidden import pandas" (exp2 iter_4 first attempt lost all 8 stats this way). Use the pre-injected `np`, `pd`, `math` with no import lines.

**Why:** a whole critique round produced zero usable statistics because of this.
**How to apply:** write stat files with no import statements; smoke-test them by exec'ing with ns={'np','pd','math'} on responses.csv.
