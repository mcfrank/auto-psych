"""LLM-as-judge structural similarity between two PyMC cognitive models.

The holdout probe measures *functional* (behavioral) equivalence — do two models
predict the same ``p_left``. This module measures the orthogonal axis: do two
models express the *same cognitive mechanism*, as read from their source code, on
a 1–7 Likert scale. An LLM judge reads both model files (blind to which is the
ground truth) and rates mechanistic similarity; walking a holdout run's
``history.json`` then yields a similarity-to-ground-truth trajectory over the
inner-loop scoring steps, the structural counterpart to the correlation
trajectory in ``holdout_recovery``.

The judge backend is injected as ``judge_fn(system, user) -> reply`` so the logic
is testable with a fake and backend-agnostic (Gemini, Claude, …). Calls are
cached by prompt hash, so a ``best_model`` that recurs across steps — or a
re-run — costs no new API calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

# A judge backend: takes (system_prompt, user_prompt), returns the raw reply.
JudgeFn = Callable[[str, str], str]

SIMILARITY_MIN = 1
SIMILARITY_MAX = 7

# Bumping this invalidates every cached judgement (the cache key includes the
# prompts verbatim, so any wording change already invalidates — this is an
# explicit extra lever). v2 adds each model's plain-English hypothesis.
PROMPT_VERSION = "v2"

SYSTEM_PROMPT = f"""\
You are a cognitive scientist comparing two computational models of \
the same behavioural task (judging which of two coin-flip sequences looks more \
random). Each model is given as a short plain-English **hypothesis** (when \
available) and the **PyMC program** that implements it. Rate how similar the two \
models are as *cognitive hypotheses* — i.e. whether they claim people use the \
**same underlying mental process** — on a 1–7 scale.

Use the stated hypothesis and the code together. If they disagree, trust what the \
code actually computes. Ignore differences in: imports, variable names, `pm.Data` plumbing, \
prior distributions, and their constants, decision temperatures, and side-bias terms. Focus on: which \
features of the sequences drive the judgement, and how those features are \
combined into a preference.

Scale anchors:
1 = entirely different mechanisms (different driving features AND different \
computation; e.g. "distance to an ideal alternation rate" vs. "length of the \
longest run").
4 = partial overlap (share one cue or one step, but the core computation \
differs; e.g. both use alternation rate, but one is a prototype distance and the \
other a Bayesian likelihood ratio).
7 = the same mechanism (same driving features combined the same way; differ at \
most in parameterisation or coding style).

Respond with a single JSON object and nothing else:
{{"rating": <integer {SIMILARITY_MIN}-{SIMILARITY_MAX}>, "rationale": "<one sentence naming the mechanism of each and why they do or don't match>"}}
"""


def build_user_prompt(
    code_a: str,
    code_b: str,
    *,
    hypothesis_a: Optional[str] = None,
    hypothesis_b: Optional[str] = None,
) -> str:
    """Render the two models as an A/B comparison (judge stays blind to which is GT).

    Each model block carries its plain-English hypothesis (when available) above
    its code, so the judge reads the stated mechanism and its implementation
    together.
    """

    def _block(label: str, code: str, hypothesis: Optional[str]) -> str:
        parts = [f"Model {label}:"]
        if hypothesis and hypothesis.strip():
            parts.append("Stated hypothesis (plain English):\n" + hypothesis.strip())
        parts.append("Code:\n```python\n" + code.strip() + "\n```")
        return "\n\n".join(parts)

    return (
        _block("A", code_a, hypothesis_a)
        + "\n\n"
        + _block("B", code_b, hypothesis_b)
        + "\n\nRate the mechanistic similarity of Model A and Model B."
    )


def load_hypothesis(model_dir: Path, name: str) -> Optional[str]:
    """The plain-English hypothesis for model ``name`` in ``model_dir``.

    Prefers a ``<name>.hypothesis.md`` file (written for inner-loop candidates);
    falls back to the model's ``rationale`` in ``models_manifest.yaml`` (carried
    by seed and exported models, and by the ground-truth seed manifest). Returns
    ``None`` if neither states a hypothesis — then only the code is judged.
    """
    import yaml

    model_dir = Path(model_dir)
    hyp_path = model_dir / f"{name}.hypothesis.md"
    if hyp_path.exists():
        text = hyp_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    manifest_path = model_dir / "models_manifest.yaml"
    if manifest_path.exists():
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        for entry in data.get("models") or []:
            if isinstance(entry, dict) and entry.get("name") == name:
                rationale = (entry.get("rationale") or "").strip()
                return rationale or None
    return None


def parse_rating(reply: str) -> int:
    """Extract the integer 1–7 rating from a judge reply.

    Tries, in order: a JSON object with a ``rating`` key, an explicit
    ``rating: N`` / ``N/7`` phrasing, then a lone integer in range. Raises
    loudly if no in-range rating is found rather than guessing a default — a
    judge that did not answer must not be silently scored.
    """
    # 1. JSON object with a "rating" field.
    for match in re.finditer(r"\{[^{}]*\}", reply, flags=re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "rating" in obj:
            value = obj["rating"]
            if isinstance(value, (int, float)) and _in_range(int(value)):
                return int(value)

    # 2. Explicit "rating: N" or "N/7".
    for pattern in (r"rating[\"']?\s*[:=]\s*([1-7])\b", r"\b([1-7])\s*/\s*7\b"):
        m = re.search(pattern, reply, flags=re.IGNORECASE)
        if m and _in_range(int(m.group(1))):
            return int(m.group(1))

    # 3. Fallback: a single standalone in-range integer.
    ints = [int(tok) for tok in re.findall(r"\b([0-9]+)\b", reply)]
    in_range = [i for i in ints if _in_range(i)]
    if len(in_range) == 1:
        return in_range[0]

    raise ValueError(
        f"Could not parse a {SIMILARITY_MIN}-{SIMILARITY_MAX} rating from judge "
        f"reply: {reply!r}"
    )


def _in_range(value: int) -> bool:
    return SIMILARITY_MIN <= value <= SIMILARITY_MAX


def _cache_key(system: str, user: str) -> str:
    digest = hashlib.sha256(f"{system}\x00{user}".encode("utf-8")).hexdigest()
    return f"{PROMPT_VERSION}:{digest}"


def _judge_once(
    system: str,
    user: str,
    *,
    judge_fn: JudgeFn,
    cache: Optional[Dict[str, Any]],
) -> int:
    """One judged rating for an ordered (system, user) prompt, cache-aware."""
    key = _cache_key(system, user)
    if cache is not None and key in cache:
        return int(cache[key]["rating"])
    reply = judge_fn(system, user)
    rating = parse_rating(reply)
    if cache is not None:
        cache[key] = {"rating": rating, "reply": reply}
    return rating


def judge_pair(
    code_a: str,
    code_b: str,
    *,
    judge_fn: JudgeFn,
    cache: Optional[Dict[str, Any]] = None,
    symmetrize: bool = True,
    hypothesis_a: Optional[str] = None,
    hypothesis_b: Optional[str] = None,
) -> Dict[str, Any]:
    """Rate mechanistic similarity of two models on the 1–7 scale.

    Each model is judged from its plain-English hypothesis (when given) plus its
    code. With ``symmetrize`` (default) the pair is judged in both orders (A,B)
    and (B,A) — swapping hypothesis and code together — and the ratings
    averaged, cancelling the judge's position bias; the returned ``similarity``
    is then a float in ``[1, 7]``. Otherwise a single ordered judgement is
    returned.
    """
    ratings = [
        _judge_once(
            SYSTEM_PROMPT,
            build_user_prompt(
                code_a, code_b, hypothesis_a=hypothesis_a, hypothesis_b=hypothesis_b
            ),
            judge_fn=judge_fn,
            cache=cache,
        )
    ]
    if symmetrize:
        ratings.append(
            _judge_once(
                SYSTEM_PROMPT,
                build_user_prompt(
                    code_b, code_a, hypothesis_a=hypothesis_b, hypothesis_b=hypothesis_a
                ),
                judge_fn=judge_fn,
                cache=cache,
            )
        )
    return {
        "similarity": sum(ratings) / len(ratings),
        "ratings": ratings,
        "symmetrized": symmetrize,
    }


def load_cache(path: Optional[Path]) -> Dict[str, Any]:
    """Load the judge cache from ``path`` (empty dict if absent)."""
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_cache(cache: Mapping[str, Any], path: Optional[Path]) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cache, indent=2), encoding="utf-8")


def similarity_trajectory(
    run_root: Path,
    gt_model: str,
    *,
    seed_models_dir: Path,
    n_experiments: int,
    judge_fn: JudgeFn,
    cache: Optional[Dict[str, Any]] = None,
    symmetrize: bool = True,
) -> List[Dict[str, Any]]:
    """Judge the then-best model vs. the ground truth at every scoring step.

    Reads each experiment's ``model_loop/history.json``, loads the then-best
    model's source from ``model_loop/models/<best>.py``, and rates it against the
    held-out ground-truth seed source. Fails loudly if a best-model file is
    missing — a step whose code cannot be read must not be silently skipped.
    """
    run_root = Path(run_root)
    seed_models_dir = Path(seed_models_dir)
    gt_path = seed_models_dir / f"{gt_model}.py"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth model source not found: {gt_path}")
    gt_code = gt_path.read_text(encoding="utf-8")
    gt_hypothesis = load_hypothesis(seed_models_dir, gt_model)

    rows: List[Dict[str, Any]] = []
    global_step = 0
    for exp_num in range(1, n_experiments + 1):
        loop_dir = run_root / f"experiment{exp_num}" / "model_loop"
        history_path = loop_dir / "history.json"
        if not history_path.exists():
            raise FileNotFoundError(
                f"No history.json for experiment {exp_num}: {history_path}"
            )
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history:
            raise ValueError(f"Empty inner-loop history: {history_path}")
        models_dir = loop_dir / "models"
        for entry in history:
            best = entry["best_model"]
            best_path = models_dir / f"{best}.py"
            if not best_path.exists():
                raise FileNotFoundError(
                    f"Best-model source missing for experiment {exp_num} "
                    f"step {entry['step']}: {best_path}"
                )
            judged = judge_pair(
                gt_code,
                best_path.read_text(encoding="utf-8"),
                judge_fn=judge_fn,
                cache=cache,
                symmetrize=symmetrize,
                hypothesis_a=gt_hypothesis,
                hypothesis_b=load_hypothesis(models_dir, best),
            )
            rows.append(
                {
                    "experiment": exp_num,
                    "step": entry["step"],
                    "iteration": entry["iteration"],
                    "global_step": global_step,
                    "best_model": best,
                    "similarity": judged["similarity"],
                    "ratings": judged["ratings"],
                }
            )
            global_step += 1
    return rows


def run_similarity(
    result: Mapping[str, Any],
    *,
    judge_fn: JudgeFn,
    seed_models_dir: Optional[Path] = None,
    cache: Optional[Dict[str, Any]] = None,
    symmetrize: bool = True,
) -> Dict[str, Any]:
    """Judge every held-out ground truth in a holdout-recovery result dict."""
    seed_models_dir = Path(seed_models_dir or result["seed_models_dir"])
    n_experiments = result["n_experiments"]
    gt_runs = []
    for gt_run in result["gt_runs"]:
        trajectory = similarity_trajectory(
            Path(gt_run["run_root"]),
            gt_run["gt_model"],
            seed_models_dir=seed_models_dir,
            n_experiments=n_experiments,
            judge_fn=judge_fn,
            cache=cache,
            symmetrize=symmetrize,
        )
        gt_runs.append({"gt_model": gt_run["gt_model"], "trajectory": trajectory})
    return {"symmetrized": symmetrize, "gt_runs": gt_runs}


def similarity_summary_text(result: Mapping[str, Any]) -> str:
    """Per-GT first/last/peak similarity as a compact text block."""
    lines = ["LLM-judged structural similarity to ground truth (1=different, 7=same)"]
    for run in result["gt_runs"]:
        traj = run["trajectory"]
        first = traj[0]["similarity"]
        last = traj[-1]["similarity"]
        peak = max(t["similarity"] for t in traj)
        lines.append(
            f"  {run['gt_model']:>26}: start {first:.1f}  end {last:.1f}  "
            f"peak {peak:.1f}  (final best: {traj[-1]['best_model']})"
        )
    return "\n".join(lines)


def plot_similarity_trajectories(
    result: Mapping[str, Any],
    out_path: Path,
    *,
    holdout_result: Optional[Mapping[str, Any]] = None,
) -> None:
    """Plot judged similarity (1–7) vs. inner-loop step, one panel per GT.

    If ``holdout_result`` is given, the per-step functional correlation
    (Pearson r with the ground truth's ``p_left``) is overlaid on a twin axis,
    so the structural-similarity line and the behavioural-recovery line can be
    read together — the gap between them is the "mimics the behaviour without
    matching the mechanism" effect.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SIM_COLOR = "#8452A0"
    CORR_COLOR = "#4878CF"

    corr_by_gt = {}
    if holdout_result is not None:
        for gt_run in holdout_result["gt_runs"]:
            corr_by_gt[gt_run["gt_model"]] = {
                row["global_step"]: row.get("pearson_r") for row in gt_run["trajectory"]
            }

    runs = result["gt_runs"]
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False, sharey=True)
    for ax, run in zip(axes[0], runs):
        traj = run["trajectory"]
        xs = [row["global_step"] for row in traj]
        ys = [row["similarity"] for row in traj]
        ax.plot(
            xs, ys, color=SIM_COLOR, marker="o", label="structural similarity (1–7)"
        )

        for x in sorted(
            row["global_step"]
            for row in traj
            if row["step"] == 0 and row["experiment"] > 1
        ):
            ax.axvline(x - 0.5, color="grey", linestyle=":", linewidth=0.8)

        ax.set_ylim(SIMILARITY_MIN - 0.5, SIMILARITY_MAX + 0.5)
        ax.set_title(run["gt_model"])
        ax.set_xlabel("inner-loop scoring step")

        corr = corr_by_gt.get(run["gt_model"])
        if corr:
            ax2 = ax.twinx()
            cpoints = [
                (row["global_step"], corr.get(row["global_step"]))
                for row in traj
                if corr.get(row["global_step"]) is not None
            ]
            if cpoints:
                cx, cy = zip(*cpoints)
                ax2.plot(
                    cx,
                    cy,
                    color=CORR_COLOR,
                    linestyle="--",
                    marker="s",
                    label="functional recovery (Pearson r)",
                )
            ax2.set_ylim(-1.05, 1.05)
            ax2.set_ylabel("Pearson r with GT p_left", color=CORR_COLOR)
            ax2.tick_params(axis="y", labelcolor=CORR_COLOR)

    axes[0][0].set_ylabel("LLM structural similarity to GT (1–7)", color=SIM_COLOR)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=SIM_COLOR, marker="o", label="structural similarity (1–7)")
    ]
    if corr_by_gt:
        handles.append(
            Line2D(
                [],
                [],
                color=CORR_COLOR,
                linestyle="--",
                marker="s",
                label="functional recovery (Pearson r)",
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8)
    title = "Structural similarity of best model to held-out ground truth"
    if corr_by_gt:
        title += " (vs. functional recovery)"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
