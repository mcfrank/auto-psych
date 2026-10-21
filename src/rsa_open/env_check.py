"""Check that this Python can run the open campaign, before any agent time is spent.

    <venv python> -m src.rsa_open.env_check

Imports every third-party package and every harness module a chain uses, and renders one SVG
(the pilot's legibility checks see materials as PNGs through resvg, a native wheel). Exits 1
with what is missing. `scripts/rsa/slurm/open_rehearsal.sbatch` runs it first: the first
rehearsal submission failed on packages the shared venv lacked (google-genai, pydantic,
resvg-py; `scripts/_venv.sh` now installs them).
"""

from __future__ import annotations

import importlib
import sys
from typing import List

THIRD_PARTY = ("jax", "numpyro", "memo", "numpy", "pandas", "scipy", "arviz", "yaml", "tyro", "pyprojroot",
               "pydantic", "google.genai", "resvg_py")
HARNESS = ("src.rsa_open.run_chain", "src.rsa_open.chain", "src.rsa_open.ground_truth", "src.rsa_open.seeds",
           "src.rsa_open.designer", "src.rsa_open.tools", "src.rsa_open.pilot", "src.rsa_open.design",
           "src.rsa_open.loop.orchestrator", "src.rsa_open.loop.critique", "src.rsa_open.loop.check_candidate")
SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8">'
       '<rect width="8" height="8" fill="#336699"/></svg>')


def problems() -> List[str]:
    out = []
    for name in THIRD_PARTY + HARNESS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # any failure to import is the environment's
            out.append(f"{name}: {type(exc).__name__}: {exc}")
    if not any(p.startswith("resvg_py") for p in out):
        try:
            import resvg_py

            png = bytes(resvg_py.svg_to_bytes(svg_string=SVG))
            if not png.startswith(b"\x89PNG"):
                out.append("resvg_py: rendering an SVG did not give a PNG")
        except Exception as exc:
            out.append(f"resvg_py: rendering an SVG failed: {type(exc).__name__}: {exc}")
    return out


def main() -> int:
    found = problems()
    if found:
        print("ERROR: this Python cannot run the open campaign (build the venv: scripts/_venv.sh "
              "build_shared_venv):\n" + "\n".join(f"  - {p}" for p in found), file=sys.stderr)
        return 1
    print(f"[env_check] OK: {len(THIRD_PARTY)} packages and {len(HARNESS)} harness modules import; resvg renders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
