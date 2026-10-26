"""The PI's review of an open chain's first paradigm (PI 2026-10-26: the first paradigm of each open
chain only), before it is published live (`src.rsa_open.live.check_review`).

    python -m src.rsa_open.review --frozen <run>/experiment1/design/frozen               # show it
    python -m src.rsa_open.review --frozen ... --preview review.html                     # a page to click through
    python -m src.rsa_open.review --frozen ... --by mcfrank                              # approve
    python -m src.rsa_open.review --frozen ... --by mcfrank --reject "the checks are ambiguous"

The usual order: a test launch (``PROLIFIC_MODE=test``) deploys the page and a Prolific draft
and stops; review the deployed page and the draft; approve; launch live. The approval names the
paradigm's sha256, so a changed paradigm needs a new one. A rejection stops the chain: what
happens next is the failure rule's (`PREREG_DRAFT.md` §4a), decided by the PI, not by this tool.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tyro

REVIEW_FILE = "review.json"


def describe(frozen: Path) -> str:
    from src.rsa_open.paradigm import load_paradigm

    p = load_paradigm(Path(frozen) / "paradigm.json")
    record = json.loads((Path(frozen) / "record.json").read_text())
    spec = json.loads((Path(frozen) / "paradigm.json").read_text())
    lines = [f"{p.name}  (sha256 {p.sha256})", "", p.summary, "",
             f"response   : {p.response.type} ({spec['response'].get('link')}); prompt: {p.response.prompt}",
             f"design     : N {p.design.n_participants} x {p.design.trials_per_participant} trials "
             f"(+ {spec['design'].get('catch_trials', 0)} catch), {p.design.minutes:g} min, ${p.design.pay_usd:.2f} each, "
             f"${p.cost_usd:,.2f} in all",
             f"submitted  : by {record.get('submitted_by', 'the designer')}"]
    if spec.get("second_response"):
        lines.append(f"second     : {spec['second_response'].get('type')}; prompt: {spec['second_response'].get('prompt')}")
    lines += ["", "welcome    : " + spec["instructions"]["welcome"], "task       : " + spec["instructions"]["task"]]
    for c in spec["instructions"].get("comprehension_checks", []):
        lines.append(f"check      : {c['question']} {c['options']} (right: {c['options'][c['correct']]})")
    pilot = record.get("pilot") or {}
    if pilot:
        lines.append("pilot      : " + ", ".join(f"{k} {pilot[k]}" for k in ("comprehension_final", "catch_pass", "readable",
                                                                            "estimated_minutes") if k in pilot))
    return "\n".join(lines)


def write_review(frozen: Path, by: str, reject: Optional[str] = None) -> Path:
    from src.rsa_open.paradigm import load_paradigm

    if not by.strip():
        raise ValueError("a review names its reviewer (--by)")
    path = Path(frozen) / REVIEW_FILE
    if path.exists():
        raise FileExistsError(f"{path} exists: a paradigm is reviewed once (move it aside to review again)")
    p = load_paradigm(Path(frozen) / "paradigm.json")
    review = {"approved": reject is None, "by": by, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "paradigm_sha256": p.sha256, **({"reason": reject} if reject else {})}
    path.write_text(json.dumps(review, indent=1))
    return path


@dataclass
class Args:
    frozen: Path
    by: Optional[str] = None
    reject: Optional[str] = None
    preview: Optional[Path] = None
    """Write a preview of the page (every list reachable with ?list=k) here."""


def main(args: Args) -> None:
    print(describe(args.frozen))
    if args.preview:
        from src.rsa_open.page import render

        lists = json.loads((Path(args.frozen) / "lists.json").read_text())
        args.preview.write_text(render(lists, preview=True), encoding="utf-8")
        print(f"\npreview -> {args.preview} (open it in a browser; add ?list=<k> for list k)", file=sys.stderr)
    if args.by:
        path = write_review(args.frozen, args.by, args.reject)
        print(f"\n{'rejected' if args.reject else 'approved'}: {path}", file=sys.stderr)
    elif args.reject:
        raise SystemExit("--reject needs --by")


if __name__ == "__main__":
    main(tyro.cli(Args))
