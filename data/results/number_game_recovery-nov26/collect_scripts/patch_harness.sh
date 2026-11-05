#!/bin/bash
# patch_harness.sh <work root suffix>: apply the export fit-alias fix to a
# sweep's staged harness_repo (src/ only), as the earlier code_patches were.
set -euo pipefail
S=/sailhome/wurgaft/auto-psych-runs/nov26b-collect
W=/scr/wurgaft/auto-psych/number_game_recovery-nov26b-$1
P=$S/export_fit_alias.src.patch
cd $W/harness_repo
for f in src/models/pymc_inference.py src/pipelines/outer_loop/model_loop_runner.py src/pipelines/outer_loop/reprune.py; do
  cmp $f $S/base_d07c467a/$f || { echo "ERROR: $f differs from d07c467a; not patching" >&2; exit 1; }
done
git apply --check $P
git apply $P
cp $P $W/code_patches/a38c7f7b+export_fit_alias.src.patch
echo "$(date -Iseconds) applied the export fit-alias fix (uncommitted, on a38c7f7b: _export_inner_loop_models files a renamed model's cached fit under its export name) on top of $(cat $W/code_commit)" >> $W/code_patches/README
echo "patched $W/harness_repo"
