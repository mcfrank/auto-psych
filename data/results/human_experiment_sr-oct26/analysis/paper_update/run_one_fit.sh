#!/bin/bash
# One paper fit: run_one_fit.sh <dataset> <model>. Own compile dir, own log.
D="$(cd "$(dirname "$0")" && pwd)"
cd "$D/../../../../.."   # repository root
tag="$1__$2"
export PYTENSOR_FLAGS="cxx=$D/clangwrap/clang++,base_compiledir=$D/ptc/$tag"
uv run python "$D/paper_fits.py" fit --dataset "$1" --model "$2" > "$D/fitlogs/$tag.log" 2>&1
status=$?
echo "[task] $tag exit $status $(grep '^\[fit\]' "$D/fitlogs/$tag.log" | tail -1)"
