"""CriticAL for the open loop: a posterior-predictive critique of the incumbent theory.

The closed loop's critic (`src.rsa.loop.critique`, PLAN.md §5: the open loop keeps the critic)
for theories that answer any paradigm through its link. Shared, unchanged: the statistic checks
(`critique_round._usable_test_statistics`), the scoring (`src.critique.ppc.score_statistics`:
two-sided empirical p, z, Benjamini-Hochberg q), main's settings (8 statistics, p <= 0.05, 1,000
replicates, one retry, "no critique" recorded) and the `critiques.md` write-up.

What is the open loop's own:

- **The rows:** the newest block, the experiment the loop selects on (the closed rule of
  2026-10-11: the critic scores the selection rows). Every earlier paradigm, and E, is held by
  the guard instead.
- **The frame** (`critique_frame`): the theorists' data rows (`brief.write_block_data`) as
  columns: the scene, the trial's design (a probed object, a speaker's target, a judged message)
  and the response in its link's own column (`OUTCOME`).
- **The replicates** (`simulate_outcomes`): each from one posterior draw of the incumbent's
  theory and this block's link parameters, through the block's link (`src.rsa_open.fit.simulate`),
  with every trial's design as observed. Only the response column differs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import jax
import numpy as np
import pandas as pd

from src.pipelines.inner_loop.critique_round import (
    CRITIQUE_N_PROPOSALS,
    CRITIQUE_PPC_REPLICATES,
    CRITIQUE_SIGNIFICANCE_ALPHA,
    CRITIQUE_STATUS_CRITIQUED,
    CRITIQUE_STATUS_NONE,
    MAX_CRITIQUE_RETRIES,
)
from src.rsa.loop.critique import CritiqueOutcome, disabled_status  # noqa: F401  (re-exported for the loop)
from src.rsa_open.corpus import Block, CorpusFit, block_draws
from src.rsa_open.fit import TheoryModel

PROMPT = Path(__file__).resolve().parent / "prompts" / "open_critique.md"
SpawnFn = Callable[[Path, str], bool]

# The response column of each link (the rest of a response is the trial's design).
OUTCOME = {"choice": "choice", "rating": "rating", "allocation": "points", "ordered_beta": "value",
           "production": "utterance", "binary_judgment": "answer"}
OUTCOME_WORDS = {
    "choice": "`choice`: the index of the object picked",
    "rating": "`rating`: the rating given to the probed object `probe` (1 = lowest)",
    "allocation": "`points`: a JSON list, the points put on each object",
    "ordered_beta": "`value`: the slider's value for the probed object `probe` (0 to its maximum)",
    "production": "`utterance`: the index of the feature whose word was chosen for the target `target`",
    "binary_judgment": "`answer`: true if the message `utterance` was judged right for the target `target`",
}


def _cell(v: Any) -> Any:
    return json.dumps(v) if isinstance(v, (list, tuple)) else v


def outcome_columns(link) -> List[str]:
    """The response columns of a block's link: one, or a two-measure link's two (the second's
    prefixed ``second_``)."""
    if link.name == "pair":
        return [OUTCOME[link.first.name], "second_" + OUTCOME[link.second.name]]
    return [OUTCOME[link.name]]


SECOND_OUTCOME_WORDS = {
    "choice": "`second_choice`: the index of the object picked in answer to the second question",
    "rating": "`second_rating`: the second question's rating of the probed object `second_probe` (1 = lowest)",
    "binary_judgment": ("`second_answer`: the second question's yes/no on the message `second_utterance` for the "
                        "target `second_target`"),
}


def outcome_words(link) -> str:
    if link.name == "pair":
        return f"{OUTCOME_WORDS[link.first.name]}; then {SECOND_OUTCOME_WORDS[link.second.name]}"
    return OUTCOME_WORDS[link.name]


def _flat(response: Mapping[str, Any]) -> Dict[str, Any]:
    """A response as columns: a two-measure response's second part as ``second_<key>``."""
    out = {k: v for k, v in response.items() if k != "second"}
    out.update({"second_" + k: v for k, v in dict(response.get("second", {})).items()})
    return out


def critique_frame(block: Block) -> pd.DataFrame:
    """One row per trial: ``trial``, ``unit``, ``source``, ``objects`` (JSON), ``heard`` (NaN when
    no word was heard), the response's design keys and its outcome column."""
    rows = []
    for i, (ctx, response, unit) in enumerate(zip(block.contexts, block.responses, block.units)):
        rows.append({"trial": i, "unit": unit, "source": block.sources[i] if block.sources else block.name,
                     "objects": json.dumps([list(r) for r in ctx.objects]),
                     "heard": np.nan if ctx.utterance is None else ctx.utterance,
                     **{k: _cell(v) for k, v in _flat(response).items()}})
    return pd.DataFrame(rows)


def simulate_outcomes(theory: TheoryModel, fitted: CorpusFit, block: Block, n_replicates: int,
                      seed: int) -> List[List[Any]]:
    """One list of simulated outcomes per replicate, each from one posterior draw (evenly spaced;
    at most as many replicates as draws), with every trial's design as observed. The theory's
    latents are computed for all replicates at once, per scene shape; only the link's sampling
    runs per replicate."""
    import jax.numpy as jnp

    from src.rsa.context import group_by_shape
    from src.rsa.fit import draw_batches

    link = block.link
    flat = block_draws(fitted.flat(), theory, block)
    total = len(next(iter(flat.values())))
    take = np.linspace(0, total - 1, min(n_replicates, total)).astype(int)
    responses = [dict(r) for r in block.responses]
    probes = np.asarray([r.get("probe", r.get("target", 0)) for r in responses])
    utterances = np.asarray([r.get("utterance", 0) for r in responses])
    pair = link.name == "pair"
    if pair:  # the second measure's design, as observed
        seconds = [dict(r["second"]) for r in responses]
        second_probes = np.asarray([r.get("probe", r.get("target", 0)) for r in seconds])
        second_utterances = np.asarray([r.get("utterance", 0) for r in seconds])
    groups = list(group_by_shape(block.contexts).values())
    theory_flat = {k: flat[k] for k in theory.params}
    latents = []
    for g in groups:
        if not theory_flat:
            latents.append(np.asarray(theory.latent_draws({}, g, link.side == "speaker", len(take))))
            continue
        parts = [np.asarray(theory.latent_draws(params, g, link.side == "speaker", n))[:n]
                 for n, params in draw_batches(theory_flat, take)]
        latents.append(np.concatenate(parts))
    columns = outcome_columns(link)
    out = []
    for r, d in enumerate(take):
        lp = {k: jnp.asarray(flat[k][d]) for k in link.params}
        row: List[Any] = [None] * block.n
        for k, (g, lat) in zip(jax.random.split(jax.random.PRNGKey(seed * 100_003 + r), len(groups)),
                               zip(groups, latents)):
            extra = ({"second_probes": second_probes[g.indices], "second_utterances": second_utterances[g.indices]}
                     if pair else {})
            sampled = link.sample(k, lat[r], lp, probes[g.indices], g.arrays, utterances[g.indices], **extra)
            for i, s in zip(g.indices, sampled):
                cells = _flat(s)
                row[i] = _cell(cells[columns[0]]) if len(columns) == 1 else tuple(_cell(cells[c]) for c in columns)
        out.append(row)
    return out


class ReplicateFrames:
    """The replicate frames, built one at a time when iterated (only the outcome column differs;
    with several ``column``s, each outcome is a tuple of their values)."""

    def __init__(self, observed: pd.DataFrame, column, outcomes: List[List[Any]]) -> None:
        if outcomes and len(outcomes[0]) != len(observed):
            raise ValueError(f"{len(outcomes[0])} simulated responses per replicate for {len(observed)} trials")
        self.observed, self.column, self.outcomes = observed, column, outcomes

    def __len__(self) -> int:
        return len(self.outcomes)

    def __iter__(self):
        for row in self.outcomes:
            df = self.observed.copy()
            if isinstance(self.column, str):
                df[self.column] = row
            else:
                for j, c in enumerate(self.column):
                    df[c] = [v[j] for v in row]
            yield df


def context_md(*, critique_dir: Path, incumbent: str, incumbent_file: Path, hypothesis: str, block: Block,
               paradigm_note: str, data_path: Path, observed: pd.DataFrame, n_proposals: int, alpha: float,
               n_replicates: int) -> str:
    from src.critique.ppc import _TEST_STAT_BUDGET_SEC, _TEST_STAT_CALL_TIMEOUT_SEC

    design = [c for c in ("probe", "target", "second_probe", "second_target") if c in observed.columns]
    if block.link.name == "binary_judgment":
        design.append("utterance")
    if "second_utterance" in observed.columns:
        design.append("second_utterance")
    n_scenes = observed[["objects", "heard"]].astype(str).drop_duplicates().shape[0]
    return f"""# Critique context

**Incumbent (best) theory:** `{incumbent}`
**Its code:** `{incumbent_file}`
**Its hypothesis:** {hypothesis.strip() or "(no stated hypothesis)"}

**The experiment critiqued:** `{block.name}`, the newest one (the loop selects theories on it).
{paradigm_note}
Its link (response process): `{block.link.name}`.

**The data your statistics receive:** {len(observed)} trials (rows) over {n_scenes} distinct scenes
(`objects` x `heard`), as in `{data_path}`, with these columns: `{", ".join(observed.columns)}`.
- `objects`: a JSON list of 0/1 feature lists, one per object.
- `heard`: the index of the feature whose word was heard (NaN when no word was heard).
- `unit`: the designed trial the row belongs to.{"" if not design else chr(10) + "- " + ", ".join(f"`{c}`" for c in design) + ": the trial's design (as presented, the same in every simulated dataset)."}
- **The response:** {outcome_words(block.link)}.

**Your job:** propose **{n_proposals}** test statistics. Write each to
`{critique_dir / "test_stats"}/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on {n_replicates} datasets simulated from the incumbent's
posterior through this experiment's link (only {", ".join(f"`{c}`" for c in outcome_columns(block.link))} differ{"s" if len(outcome_columns(block.link)) == 1 else ""} between them),
and flags a statistic as a **significant discrepancy** when its two-sided p ≤ {alpha}
(with a Benjamini–Hochberg q beside it).

**Time:** each statistic is called {n_replicates + 1} times. Each call must finish
within {_TEST_STAT_CALL_TIMEOUT_SEC:g} s and all of them within {_TEST_STAT_BUDGET_SEC:g} s
(about {_TEST_STAT_BUDGET_SEC / (n_replicates + 1):.2g} s per call on {len(observed)} rows).
Compute scene properties once per distinct scene and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
"""


def build_prompt(critique_dir: Path, context_text: str, *, attempt: int = 0,
                 broken: Optional[Dict[str, str]] = None) -> str:
    from src.rsa.loop import critique as closed

    text = closed.build_prompt(critique_dir, context_text, attempt=attempt, broken=broken)
    old = closed.PROMPT.read_text(encoding="utf-8")
    if not text.startswith(old):
        raise RuntimeError("the closed critique prompt changed shape; update src.rsa_open.loop.critique")
    return PROMPT.read_text(encoding="utf-8") + text[len(old):]


def run_critique(round_dir: Path, *, spawn: SpawnFn, incumbent: str, theory: TheoryModel, fitted: CorpusFit,
                 hypothesis: str, incumbent_file: Path, block: Block, paradigm_note: str, data_path: Path,
                 n_proposals: int = CRITIQUE_N_PROPOSALS, alpha: float = CRITIQUE_SIGNIFICANCE_ALPHA,
                 n_replicates: int = CRITIQUE_PPC_REPLICATES, seed: int = 0) -> CritiqueOutcome:
    """Critique the incumbent on ``block`` before a candidate round (the closed critic's rules: one
    retry; a round whose statistics all fail, or none are written, has no critique, recorded)."""
    from src.critique.ppc import score_statistics
    from src.pipelines.inner_loop.critique_round import _format_critiques_md, _usable_test_statistics

    observed = critique_frame(block)
    broken: Dict[str, str] = {}
    usable: List[Path] = []
    cdir = round_dir / "critique"
    attempts = 0
    for attempt in range(1 + MAX_CRITIQUE_RETRIES):
        attempts = attempt + 1
        cdir = round_dir / ("critique" if attempt == 0 else f"critique_retry_{attempt}")
        cdir.mkdir(parents=True, exist_ok=True)
        text = context_md(critique_dir=cdir, incumbent=incumbent, incumbent_file=incumbent_file,
                          hypothesis=hypothesis, block=block, paradigm_note=paradigm_note, data_path=data_path,
                          observed=observed, n_proposals=n_proposals, alpha=alpha, n_replicates=n_replicates)
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
    columns = outcome_columns(block.link)
    reps = ReplicateFrames(observed, columns[0] if len(columns) == 1 else columns,
                           simulate_outcomes(theory, fitted, block, n_replicates, seed))
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
        status=CRITIQUE_STATUS_CRITIQUED, incumbent=incumbent, attempts=attempts, dir=str(cdir), block=block.name,
        n_statistics=int(result["n_test_statistics"]), n_evaluated=n_evaluated,
        n_significant=int(result["n_significant"]), n_significant_fdr=int(result["n_significant_fdr"])))
