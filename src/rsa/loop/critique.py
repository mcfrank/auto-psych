"""CriticAL for the RSA loop: a posterior-predictive critique of the incumbent.

Main's inner loop critiques its incumbent before every candidate round
(`src.pipelines.inner_loop.critique_round`); this is the same step for memo
models (PI 2026-10-10: the framework should be one framework across
phenomena). Shared with main, unchanged:

- the statistic checks (import gate, one trial run, the time budget):
  `critique_round._usable_test_statistics`;
- the scoring (two-sided empirical p, z, Benjamini-Hochberg q):
  `src.critique.ppc.score_statistics`;
- the agent's retry once, "no critique" recorded when it writes nothing usable;
- the `critiques.md` write-up the candidates get.

What is RSA's own:

- **The frame** (`critique_frame`): the loop's trials (included forced-choice
  rows), only the design columns plus ``choice`` (`CRITIQUE_COLUMNS`). The table's
  other outcome columns (``response``, the click's ``display_order``) are left
  out: a replicate redraws only ``choice``, so a statistic reading them would
  compare the data with themselves. ``choice`` is the chosen *class* (identical
  objects are one choice, recorded as the first one's index), as the fit sees it.
- **The replicates** (`ReplicateFrames`): choices simulated from the
  incumbent's posterior, one posterior draw per replicate, built one frame at
  a time (1,000 copies of ~50k trials would not fit in memory at once).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Main's settings: 8 statistics a round, p <= 0.05 flags a discrepancy (BH q
# beside it), 1,000 replicates, one retry.
from src.pipelines.inner_loop.critique_round import (
    CRITIQUE_N_PROPOSALS,
    CRITIQUE_PPC_REPLICATES,
    CRITIQUE_SIGNIFICANCE_ALPHA,
    CRITIQUE_STATUS_CRITIQUED,
    CRITIQUE_STATUS_DISABLED,
    CRITIQUE_STATUS_NONE,
    MAX_CRITIQUE_RETRIES,
)
from src.rsa.context import Context, group_by_shape, unique_contexts
from src.rsa.fit import RSAFit, class_probs, draw_batches, posterior_flat
from src.rsa.model_file import RSAModel

PROMPT = Path(__file__).resolve().parent / "prompts" / "rsa_critique.md"
REPLICATE_CHUNK = 50  # replicates sampled at once (memory ~ chunk x trials x 4 x 8 bytes)

# What a statistic may read: the display, the trial's bookkeeping, and the choice.
CRITIQUE_COLUMNS = (
    "source", "experiment", "condition", "participant_id", "trial_index", "item", "feature_names",
    "objects", "object_roles", "messages", "query", "utterance", "framing", "familiarization",
    "grayscale", "choice",
)

SpawnFn = Callable[[Path, str], bool]


def critique_frame(frame: pd.DataFrame, contexts: Sequence[Context], choices: Sequence[int]) -> pd.DataFrame:
    """The observed frame: design columns plus the chosen class (`CRITIQUE_COLUMNS`)."""
    cols = [c for c in CRITIQUE_COLUMNS if c in frame.columns]
    out = frame[cols].reset_index(drop=True).copy()
    out["choice"] = [ctx.choice_classes()[int(c)] for ctx, c in zip(contexts, choices)]
    return out


def simulate_choices(model: RSAModel, fitted: RSAFit, contexts: Sequence[Context], n_replicates: int,
                     seed: int) -> np.ndarray:
    """(replicates, trials) chosen classes, each replicate from one posterior draw
    (evenly spaced over the draws; at most as many replicates as draws)."""
    flat = posterior_flat(fitted)
    total = len(next(iter(flat.values())))
    take = np.linspace(0, total - 1, min(n_replicates, total)).astype(int)
    uniques, inverse = unique_contexts(contexts)
    width = max(len(c.objects) for c in uniques)
    probs = np.zeros((len(take), len(uniques), width))
    for group in group_by_shape(uniques).values():
        start = 0
        for n, params in draw_batches(flat, take):
            p = np.asarray(class_probs(model.draws_probs(params, group), group.classes), dtype=np.float64)[:n]
            probs[start:start + n, group.indices, : group.shape[0]] = p
            start += n
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{model.name}: choice probabilities are not finite on some trials")
    rng = np.random.default_rng(seed)
    out = np.empty((len(take), len(inverse)), dtype=np.int16)
    for r0 in range(0, len(take), REPLICATE_CHUNK):
        cum = probs[r0:r0 + REPLICATE_CHUNK][:, inverse, :].cumsum(-1)  # (chunk, trials, width)
        u = rng.random(cum.shape[:2]) * cum[..., -1]
        # The first slot whose cumulative mass reaches u: always a class slot (the
        # others carry no mass), so the pick is a class's first object.
        out[r0:r0 + REPLICATE_CHUNK] = (cum < u[..., None]).sum(-1)
    return out


class ReplicateFrames:
    """The replicate frames, built one at a time when iterated."""

    def __init__(self, observed: pd.DataFrame, choices: np.ndarray) -> None:
        if choices.shape[1] != len(observed):
            raise ValueError(f"{choices.shape[1]} simulated choices per replicate for {len(observed)} trials")
        self.observed = observed
        self.choices = choices

    def __len__(self) -> int:
        return len(self.choices)

    def __iter__(self):
        for row in self.choices:
            df = self.observed.copy()
            df["choice"] = row.astype(np.int64)
            yield df


def context_md(*, critique_dir: Path, incumbent: str, incumbent_file: Path, hypothesis: str,
               responses_path: Path, observed: pd.DataFrame, n_proposals: int, alpha: float,
               n_replicates: int, scope_note: str = "") -> str:
    from src.critique.ppc import _TEST_STAT_BUDGET_SEC, _TEST_STAT_CALL_TIMEOUT_SEC

    n_displays = observed[["objects", "utterance"]].astype(str).drop_duplicates().shape[0]
    sources = ", ".join(sorted(observed["source"].dropna().unique())) if "source" in observed else "one source"
    return f"""# Critique context

**Incumbent (best) model:** `{incumbent}`
**Its code:** `{incumbent_file}`
**Its hypothesis:** {hypothesis.strip() or "(no stated hypothesis)"}

**The data your statistics receive:** {len(observed)} trials (rows) over {n_displays}
distinct displays (`objects` × `utterance`), from {sources}. The responses CSV
(`{responses_path}`) holds more columns and rows; your statistic receives only
the included forced-choice trials and these columns: `{", ".join(observed.columns)}`.
- `objects`: a JSON list of 0/1 feature lists, one per object.
- `query`: `utterance` or `prior` (a mumble trial: no word heard, `utterance` is NaN).
- `utterance`: the index of the heard feature.
- `object_roles`: where known, a JSON list naming each object's role in the original
  design (e.g. target, logical, foil).
- `choice`: the clicked object's index. Identical objects are one choice, recorded
  as the first one's index, and so is every model's simulated choice.
{scope_note}
**Your job:** propose **{n_proposals}** test statistics. Write each to
`{critique_dir / "test_stats"}/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on {n_replicates} datasets simulated from the incumbent's
posterior (only `choice` differs between them), and flags a statistic as a
**significant discrepancy** when its two-sided p ≤ {alpha} (with a Benjamini–Hochberg q
beside it).

**Time:** each statistic is called {n_replicates + 1} times. Each call must finish
within {_TEST_STAT_CALL_TIMEOUT_SEC:g} s and all of them within {_TEST_STAT_BUDGET_SEC:g} s
(about {_TEST_STAT_BUDGET_SEC / (n_replicates + 1):.2g} s per call on {len(observed)} rows).
Compute display properties once per distinct display and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
"""


def build_prompt(critique_dir: Path, context_text: str, *, attempt: int = 0,
                 broken: Optional[Dict[str, str]] = None) -> str:
    stats = critique_dir / "test_stats"
    sections = [
        PROMPT.read_text(encoding="utf-8"),
        "---",
        f"Your working directory for this critique is `{critique_dir}`. Write your test "
        f"statistics into `{stats}/` (one `test_statistic(df)` per file). Your shell runs "
        "from the repository root, so always use the absolute path. Use the bash tool "
        "with a heredoc for each file, e.g.\n"
        f"```bash\nmkdir -p {stats}\ncat << 'EOF' > {stats}/my_statistic.py\n"
        "def test_statistic(df): ...\nEOF\n```",
    ]
    if attempt > 0:
        note = (f"NOTE: this is attempt {attempt + 1}. The previous attempt left no usable test "
                f"statistic in `{stats}/`. If this one does not either, the round runs without a "
                "critique, so write the statistic files first.")
        if broken:
            note += ("\n\nThese statistics were set aside because they failed when run once on the "
                     "observed data:\n" + "\n".join(f"- {k}: {v}" for k, v in sorted(broken.items())))
        sections.append(note)
    sections.append(f"## CRITIQUE_CONTEXT.md\n\n{context_text}")
    return "\n\n".join(sections) + "\n"


@dataclass(frozen=True)
class CritiqueOutcome:
    critiques_md: Optional[str]  # the text for the candidates, or None (no critique)
    status: Dict[str, Any]  # recorded in the round's history.json entry


def run_critique(round_dir: Path, *, spawn: SpawnFn, incumbent: str, model: RSAModel, fitted: RSAFit,
                 hypothesis: str, incumbent_file: Path, frame: pd.DataFrame, contexts: Sequence[Context],
                 choices: Sequence[int], responses_path: Path, n_proposals: int = CRITIQUE_N_PROPOSALS,
                 alpha: float = CRITIQUE_SIGNIFICANCE_ALPHA, n_replicates: int = CRITIQUE_PPC_REPLICATES,
                 seed: int = 0, scope_note: str = "") -> CritiqueOutcome:
    """Critique the incumbent before a candidate round (main's rules: one retry;
    a round whose statistics all fail, or none are written, has no critique,
    recorded). The first attempt is ``round_dir/critique``, a retry
    ``critique_retry_<k>``; each attempt is its own agent directory and log."""
    from src.critique.ppc import score_statistics
    from src.pipelines.inner_loop.critique_round import _format_critiques_md, _usable_test_statistics

    observed = critique_frame(frame, contexts, choices)
    broken: Dict[str, str] = {}
    usable: List[Path] = []
    cdir = round_dir / "critique"
    attempts = 0
    for attempt in range(1 + MAX_CRITIQUE_RETRIES):
        attempts = attempt + 1
        cdir = round_dir / ("critique" if attempt == 0 else f"critique_retry_{attempt}")
        cdir.mkdir(parents=True, exist_ok=True)
        text = context_md(critique_dir=cdir, incumbent=incumbent, incumbent_file=incumbent_file,
                          hypothesis=hypothesis, responses_path=responses_path, observed=observed,
                          n_proposals=n_proposals, alpha=alpha, n_replicates=n_replicates, scope_note=scope_note)
        (cdir / "CRITIQUE_CONTEXT.md").write_text(text, encoding="utf-8")
        spawn(cdir, build_prompt(cdir, text, attempt=attempt, broken=broken))
        usable, broken = _usable_test_statistics(cdir / "test_stats", observed, n_replicates=n_replicates)
        if usable:
            break
        print(f"  [critique] attempt {attempts}: no usable test statistic in {cdir / 'test_stats'}", flush=True)
    if not usable:
        reason = f"the critique agent wrote no usable test statistic in {attempts} attempts"
        print(f"  [critique] NO CRITIQUE this round: {reason}", flush=True)
        return CritiqueOutcome(None, dict(status=CRITIQUE_STATUS_NONE, incumbent=incumbent, attempts=attempts,
                                          reason=reason, broken=broken))
    reps = ReplicateFrames(observed, simulate_choices(model, fitted, contexts, n_replicates, seed))
    result = score_statistics(incumbent, observed, reps, usable, significance_alpha=alpha)
    result["n_replicates"] = len(reps)
    (cdir / "ppc_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = _format_critiques_md(result)
    (cdir / "critiques.md").write_text(md, encoding="utf-8")
    n_evaluated = sum(1 for r in result["results"] if not r.get("error"))
    print(f"  [critique] {result['n_significant']} of {n_evaluated} evaluated statistics show a significant "
          "discrepancy", flush=True)
    if n_evaluated == 0:
        reason = ("none of the test statistics produced a p-value: "
                  + "; ".join(f"{r['name']}: {r['error']}" for r in result["results"]))
        return CritiqueOutcome(None, dict(status=CRITIQUE_STATUS_NONE, incumbent=incumbent, attempts=attempts,
                                          reason=reason))
    return CritiqueOutcome(md, dict(
        status=CRITIQUE_STATUS_CRITIQUED, incumbent=incumbent, attempts=attempts, dir=str(cdir),
        n_statistics=int(result["n_test_statistics"]), n_evaluated=n_evaluated,
        n_significant=int(result["n_significant"]), n_significant_fdr=int(result["n_significant_fdr"])))


def disabled_status() -> Dict[str, Any]:
    return dict(status=CRITIQUE_STATUS_DISABLED)
