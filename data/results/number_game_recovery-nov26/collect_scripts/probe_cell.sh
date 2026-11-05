#!/bin/bash
# probe_cell.sh <WORK_ROOT> <cell rel dir>: prints "TRIGGER ..." once experiment 2's
# model loop validates and experiment 3 has begun; else "WAIT ...". Runs on the cell's node.
set -euo pipefail
W=$1; RUN=$W/$2
E2=$(ls -d "$RUN"/repo/_runs/*/experiment2)
E3=${E2%2}3
cd "$W/harness_repo"
v=$("$W/venv/bin/python" -c "
import sys; from pathlib import Path
from src.pipelines.outer_loop.orchestrator_validators import validate_cc_output
ok, msg = validate_cc_output('5_model_loop', Path(sys.argv[1])); print(int(ok), msg[:150])" "$E2" 2>/dev/null | tail -1)
if [[ -d "$E3" && "${v%% *}" == 1 ]]; then
  echo "TRIGGER exp3: $(ls "$E3" | tr '\n' ' ')$(ls -d "$E3"/model_loop/iter_* 2>/dev/null | wc -l) iter(s)"
else
  echo "WAIT exp2 model loop valid=${v%% *} exp3_exists=$([[ -d $E3 ]] && echo 1 || echo 0) (${v#* })"
fi
