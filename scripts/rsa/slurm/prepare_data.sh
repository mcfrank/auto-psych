#!/bin/bash
# RSA run 1, step 1 -- on a Sherlock LOGIN node (it downloads; compute nodes
# need not have the data sources' hosts reachable):
#
#   cd ~/auto-psych && git checkout auto-rsa && git pull
#   bash scripts/rsa/slurm/prepare_data.sh
#
# 1. builds the sweep's venv ($WORK_ROOT/venv) from uv.lock, wheels only;
# 2. fetches the external sources without a licence from their pinned URLs
#    into $REPO/data/rsa/external (sha256-checked by the ingest itself) and
#    rebuilds the derived CSVs -- the committed ones must come out identical;
# 3. combines the five sources into $DATA_ROOT/combined_trials.csv;
# 4. splits it (held-out conditions within papers, seed $SPLIT_SEED) into
#    $DATA_ROOT/real/{train.csv,test.csv,split.json};
# 5. prints and records sha256s ($DATA_ROOT/SHA256SUMS) and the code that
#    prepared the data ($DATA_ROOT/prepared_code; setup refuses other code).
#
# Re-running is safe until the sweep starts: it refuses once $CELLS_ROOT has
# cells, and it deletes the recovery datasets (setup re-simulates them from
# the new split).
#
# VENV_MODE=sync: `uv sync --locked --no-build` -- the lock exactly,
#   failing (not building from source) if a package has no glibc-2.17 wheel.
# VENV_MODE=wheels (default; on Sherlock the locked sync fails: contourpy
# 1.3.3 has no glibc-2.17 wheel): the subjective-randomness recipe (uv pip install
#   --only-binary) with the RSA stack pinned to uv.lock's versions. Use it if
#   the sync fails on a wheel.
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
source "$REPO/scripts/rsa/slurm/_env.sh"   # cds to $REPO
SPLIT_SEED="${SPLIT_SEED:-0}"
EXTERNAL_SOURCES=(mayn_demberg_2026 mayn_demberg_2023 mayn_demberg_2022 sikos_2021)
ALL_SOURCES=(pragmods "${EXTERNAL_SOURCES[@]}")

if compgen -G "$CELLS_ROOT/*/" >/dev/null; then
  echo "ERROR: $CELLS_ROOT has cells: the sweep has started on the data in $DATA_ROOT; refusing to rebuild it. Use a new WORK_ROOT." >&2
  exit 1
fi
# (`cd`, not `git -C`: el7's git 1.8 predates -C.)
[[ "$(cd "$REPO" && git rev-parse --abbrev-ref HEAD)" == "auto-rsa" ]] \
  || echo "WARNING: $REPO is not on branch auto-rsa"
rsa_require_opencode
[[ -n "${GOOGLE_API_KEY:-}" ]] || echo "WARNING: GOOGLE_API_KEY is not set; setup and the loop will refuse to start without it"

# --- 1. venv -------------------------------------------------------------------
echo "[prepare] venv at $UV_PROJECT_ENVIRONMENT (python $UV_PYTHON, mode ${VENV_MODE:-wheels})"
case "${VENV_MODE:-wheels}" in
  sync)
    uv sync --locked --no-build ;;
  wheels)
    [[ -x "$VENV_PY" ]] || uv venv --python "$UV_PYTHON" "$UV_PROJECT_ENVIRONMENT"
    uv pip install --python "$UV_PROJECT_ENVIRONMENT" --only-binary :all: \
      "pymc==5.28.5" "pytensor==2.38.3" "arviz==0.23.4" "h5py<3.15" \
      "memo-lang==1.3.0" "jax==0.7.0" "jaxlib==0.7.0" "numpyro==0.19.0" \
      "numpy<2.3" "scipy<1.15" "pandas<2.3" \
      pyyaml tyro pyprojroot matplotlib requests threadpoolctl ;;
  *) echo "ERROR: VENV_MODE must be sync or wheels" >&2; exit 1 ;;
esac
"$VENV_PY" - <<'PY'
import sys
import jax, jax.numpy as jnp, numpyro, memo, pymc, arviz, pandas  # noqa: F401
assert sys.version_info[:2] == (3, 12), sys.version
assert jax.__version__ == "0.7.0", jax.__version__
print(f"[prepare] venv OK: python {sys.version.split()[0]}, jax {jax.__version__}, "
      f"numpyro {numpyro.__version__}, pymc {pymc.__version__}, arviz {arviz.__version__}; "
      f"jnp check {float(jnp.ones(3).sum())}")
PY

# --- 2. fetch + rebuild ------------------------------------------------------
# Run exactly as the committed provenance records it (no --cache-dir: its
# generation_command would change), so the committed CSVs and provenance
# files come out byte-identical and the checkout stays clean.
code_before=$(bash "$REPO/scripts/subjective_randomness/slurm/code_commit.sh" "$REPO")
"$VENV_PY" -m src.rsa.ingest.run --sources "${EXTERNAL_SOURCES[@]}"
code_after=$(bash "$REPO/scripts/subjective_randomness/slurm/code_commit.sh" "$REPO")
if [[ "$code_after" != "$code_before" ]]; then
  echo "ERROR: the ingest changed the checkout ($code_before -> $code_after): a committed CSV or provenance did not rebuild identically." >&2
  git status --short >&2 || true
  echo "       Investigate (a source changed upstream?); restore with: git checkout -- src/pipelines/outer_loop/projects/rsa_reference/data" >&2
  exit 1
fi

# --- 3-4. combine + split ----------------------------------------------------
mkdir -p "$DATA_ROOT"
"$VENV_PY" -m src.rsa.ingest.combine --sources "${ALL_SOURCES[@]}" --out "$DATA_ROOT/combined_trials.csv"
rm -rf "$DATA_ROOT/real" "$DATA_ROOT"/recovery_*   # derived from the old split, if any
"$VENV_PY" -m src.rsa.split --trials "$DATA_ROOT/combined_trials.csv" --out-dir "$DATA_ROOT/real" --seed "$SPLIT_SEED"

# --- 5. record ----------------------------------------------------------------
printf '%s\n' "$code_after" > "$DATA_ROOT/prepared_code"
(cd "$DATA_ROOT" && sha256sum combined_trials.csv real/train.csv real/test.csv real/split.json > SHA256SUMS)
echo "[prepare] code: $code_after"
echo "[prepare] sha256 ($DATA_ROOT/SHA256SUMS):"
cat "$DATA_ROOT/SHA256SUMS"
"$VENV_PY" - "$DATA_ROOT/real/split.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
for src, r in info["sources"].items():
    print(f"[prepare] {src}: test holds {r['held_out_units']}/{r['units']} units, "
          f"{r['held_out_trials']}/{r['trials']} trials")
PY
echo "[prepare] done. Next: bash scripts/rsa/slurm/submit.sh"
