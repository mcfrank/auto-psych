#!/bin/bash
# probe_experiment.sh <WORK_ROOT> <cell rel dir>: prints "STATE <n> <prev_valid> <iters>"
# for the cell's newest experiment n: whether experiment n-1's model loop validates
# (1 for n=1) and how many model-loop rounds experiment n has. Runs on the cell's node.
set -euo pipefail
W=$1; RUN=$W/$2
E=$(ls -d "$RUN"/repo/_runs/*/experiment[0-9]* | sort -V | tail -1)
n=${E##*experiment}
v=1
if (( n > 1 )); then
  cd "$W/harness_repo"
  v=$("$W/venv/bin/python" -c "
import sys; from pathlib import Path
from src.pipelines.outer_loop.orchestrator_validators import validate_cc_output
print(int(validate_cc_output('5_model_loop', Path(sys.argv[1]))[0]))" "${E%$n}$((n - 1))" 2>/dev/null | tail -1)
fi
echo "STATE $n $v $(ls -d "$E"/model_loop/iter_* 2>/dev/null | grep -cv _retry_)"
