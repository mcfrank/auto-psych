"""One recovery-improvement iteration must (1) digest the sweeps it is asked to
review, (2) clone the subject repo onto a fresh per-iteration branch, (3) hand
the review agent a prompt carrying the digest and the deliverable contract,
(4) validate the agent's deliverables — a prescription, committed changes, and
exactly one of ``next_run.env`` / ``STOP`` — repairing once with the problems
injected and otherwise failing loudly, (5) launch the declared sweep from the
iteration's repo, and (6) chain the next review job on the sweep's analysis
job. State crosses iterations only through on-disk artifacts, so an iteration
is fully described by its directory.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from src.recovery_improvement.campaign import Campaign
from src.recovery_improvement.iteration import SweepJobs, run_iteration

GT_MODELS = ["falk_konold_dp", "motif_stack"]
GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=GIT_ENV, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.strip()


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    """A committed subject repo with the config the sweep launcher validates."""
    repo = tmp_path / "source_repo"
    config = repo / "scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "gt_models:\n" + "".join(f"  {gt}:\n" for gt in GT_MODELS)
        + "seed_models_dir: src/subjective_randomness/pymc_model_families\n",
        encoding="utf-8",
    )
    (repo / "src").mkdir()
    (repo / "src" / "loop.py").write_text("THRESHOLD = 0.02\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _write_sweep(root: Path, *, runs: int, r_final: dict[str, list[float]]) -> None:
    """A finished sweep: aggregate JSON, per-(run, gt) csv/json, slurm logs."""
    root.mkdir(parents=True)
    per_gt = {}
    for gt in GT_MODELS:
        values = r_final[gt]
        per_gt[gt] = {
            "n_runs": len(values), "mean": sum(values) / len(values), "sd": 0.01,
            "cv": 0.01, "min": min(values), "max": max(values), "values": values,
            "modal_best_model": f"best_for_{gt}", "best_model_agreement": 0.5,
        }
    (root / "test_retest.json").write_text(json.dumps({
        "runs_root": str(root), "metric": "pearson_r",
        "runs_found": [f"run{i}" for i in range(1, runs + 1)],
        "gt_models": GT_MODELS, "runs_in_complete_matrix": ["run1"],
        "icc_2_1": 0.9, "mean_pairwise_corr": 0.8, "per_gt_model": per_gt,
    }), encoding="utf-8")
    logs = root / "slurm_logs"
    logs.mkdir()
    task = 0
    for run in range(1, runs + 1):
        for gt in GT_MODELS:
            task += 1
            gt_dir = root / f"run{run}" / gt
            gt_dir.mkdir(parents=True)
            final = r_final[gt][run - 1]
            (gt_dir / "holdout.csv").write_text(
                "gt_model,experiment,step,iteration,global_step,best_model,pearson_r,rmse\n"
                f"{gt},1,0,,0,seed_model,0.5,0.2\n"
                f"{gt},1,1,0,1,first_candidate,0.7,0.15\n"
                f"{gt},2,0,,2,best_for_{gt},{final},0.1\n",
                encoding="utf-8",
            )
            (gt_dir / "holdout.json").write_text(json.dumps({
                "n_experiments": 2, "n_participants": 40,
                "inner_loop": {"max_iterations": 2, "candidate_count": 3},
                "fit_kwargs": {"draws": 2000, "tune": 1000, "chains": 4},
                "gt_runs": [{"gt_model": gt, "leakage": {
                    "any_identical": False, "any_mention": gt == "motif_stack",
                    "any_value_mention": False, "any_gt_named": False}}],
            }), encoding="utf-8")
            (logs / f"holdout_recovery_100_{task}.out").write_text(
                f"[task {task}] repeat={run} gt={gt} seed=10{run} -> {gt_dir}\n"
                "  [tokens] holdout recovery: 100 tokens over 2 LLM call(s)  cost=$1.50\n"
                f"[task {task}] done -> {gt_dir}/holdout.json\n",
                encoding="utf-8",
            )
    # One extra task that crashed (its cell is missing above).
    (logs / f"holdout_recovery_100_{task + 1}.out").write_text(
        f"[task {task + 1}] repeat=9 gt=motif_stack seed=109 -> {root}/run9/motif_stack\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "ValueError: Stacking weights sum to 0.0; cannot form a model prior\n",
        encoding="utf-8",
    )


@pytest.fixture
def baseline_sweep(tmp_path: Path) -> Path:
    root = tmp_path / "sweeps" / "baseline_32eig_32random"
    _write_sweep(root, runs=2, r_final={"falk_konold_dp": [0.98, 0.97], "motif_stack": [0.85, 0.83]})
    return root


@pytest.fixture
def campaign(tmp_path: Path, source_repo: Path, baseline_sweep: Path) -> Campaign:
    root = tmp_path / "campaign"
    root.mkdir()
    base_branch = _git(source_repo, "rev-parse", "--abbrev-ref", "HEAD")
    (root / "campaign.env").write_text(
        f"CAMPAIGN_NAME=demo\nSOURCE_REPO={source_repo}\nBASE_BRANCH={base_branch}\n"
        f'BASELINE_ROOTS="{baseline_sweep}"\nMAX_ITERATIONS=2\n'
        "REVIEW_MODEL=claude-test\nREVIEW_MAX_TURNS=5\nREVIEW_MAX_BUDGET_USD=1\n"
        "REVIEW_TIMEOUT_SEC=60\nMAX_REVIEW_REPAIRS=1\n"
        'SWEEP_N_REPEATS=5\nSWEEP_BASE_SEED=100\nSWEEP_MAX_PARALLEL=5\n'
        f'SWEEP_GT_MODELS="{" ".join(GT_MODELS)}"\n'
        "SWEEP_CONFIG=scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml\n"
        "SWEEP_SEED_MODELS_REL=src/subjective_randomness/pymc_model_families\n",
        encoding="utf-8",
    )
    return Campaign.load(root)


PROMPT_TEMPLATE = (
    "iteration $iteration of $max_iterations for $campaign_name\n"
    "repo=$repo branch=$branch iter_dir=$iter_dir\n"
    "defaults: $sweep_defaults\n"
    "$repair_feedback\n---DIGEST---\n$digest\n---PREVIOUS---\n$previous_prescriptions\n"
    "---JOURNAL---\n$journal\n"
)


class FakeSlurm:
    """Records what the iteration asked Slurm to do and hands back job ids."""

    def __init__(self) -> None:
        self.sweeps: list[tuple[Path, dict]] = []
        self.reviews: list[tuple[int, str | None, str]] = []

    def submit_sweep(self, repo: Path, env: dict) -> SweepJobs:
        self.sweeps.append((repo, env))
        return SweepJobs(setup_id="11", array_id="12", analysis_id="13")

    def submit_review(self, iteration: int, after_job_id: str | None, mode: str) -> str:
        self.reviews.append((iteration, after_job_id, mode))
        return "14"


def _good_agent(prompts: list[str]):
    """An agent that does everything the contract asks for."""

    def run(prompt: str, *, cwd: Path, log_path: Path) -> tuple[bool, str]:
        prompts.append(prompt)
        iter_dir = cwd.parent
        with (cwd / "src" / "loop.py").open("a", encoding="utf-8") as fh:
            fh.write("THRESHOLD += 0.01  # one tweak per iteration\n")
        _git(cwd, "commit", "-q", "-am", "raise the novelty threshold")
        (iter_dir / "prescription.md").write_text("# Findings\nraise threshold\n", encoding="utf-8")
        (iter_dir / "next_run.env").write_text(
            "N_REPEATS=2\nINNER_LOOP_ITERATIONS=3\nNOTE=test the raised threshold\n",
            encoding="utf-8",
        )
        log_path.write_text("agent log", encoding="utf-8")
        return True, "done"

    return run


def test_iteration_reviews_then_launches_sweep_and_chains_next_review(campaign, baseline_sweep):
    slurm = FakeSlurm()
    prompts: list[str] = []

    result = run_iteration(
        campaign, 1, run_agent=_good_agent(prompts), submit_sweep=slurm.submit_sweep,
        submit_review=slurm.submit_review, prompt_template=PROMPT_TEMPLATE,
    )

    iter_dir = campaign.root / "iter1"
    digest = (iter_dir / "digest.md").read_text(encoding="utf-8")
    assert "falk_konold_dp" in digest and "motif_stack" in digest
    assert "Stacking weights sum to 0.0" in digest  # the crashed task is surfaced
    assert "0.98" in digest  # a per-run final value

    assert len(prompts) == 1
    assert "iteration 1 of 2 for demo" in prompts[0]
    assert digest in prompts[0]
    assert str(iter_dir / "repo") in prompts[0]

    repo = iter_dir / "repo"
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "recovery-improvement/demo/iter1"
    assert "raise the novelty threshold" in _git(repo, "log", "--oneline")

    assert len(slurm.sweeps) == 1
    launched_repo, env = slurm.sweeps[0]
    assert launched_repo == repo
    assert env["REPO"] == str(repo)
    assert env["WORK_ROOT"] == str(iter_dir / "sweep")
    assert env["N_REPEATS"] == "2" and env["INNER_LOOP_ITERATIONS"] == "3"
    assert env["BASE_SEED"] == "100" and env["GT_MODELS"] == " ".join(GT_MODELS)
    assert env["CONFIG"].endswith("holdout_recovery_faithful.yaml")
    assert "NOTE" not in env

    assert slurm.reviews == [(2, "13", "review")]
    jobs = json.loads((iter_dir / "jobs.json").read_text(encoding="utf-8"))
    assert jobs["sweep_jobs"]["analysis_id"] == "13" and jobs["next_review_job"] == "14"
    assert result.decision == "sweep" and result.repairs_used == 0

    journal = (campaign.root / "journal.md").read_text(encoding="utf-8")
    assert "Iteration 1" in journal and "13" in journal and "14" in journal


def test_final_iteration_chains_a_finalize_job_instead_of_another_review(campaign):
    slurm = FakeSlurm()
    run_iteration(
        campaign, 1, run_agent=_good_agent([]), submit_sweep=slurm.submit_sweep,
        submit_review=slurm.submit_review, prompt_template=PROMPT_TEMPLATE,
    )
    run_iteration(
        campaign, 2, run_agent=_good_agent([]), submit_sweep=slurm.submit_sweep,
        submit_review=slurm.submit_review, prompt_template=PROMPT_TEMPLATE,
    )
    assert slurm.reviews[-1] == (3, "13", "finalize")
    # Iteration 2 builds on iteration 1's branch, so the tweaks accumulate.
    log = _git(campaign.root / "iter2" / "repo", "log", "--oneline")
    assert log.count("raise the novelty threshold") == 2
    digest2 = (campaign.root / "iter2" / "digest.md").read_text(encoding="utf-8")
    assert "iter1" in digest2  # the previous sweep is reviewed (even if unfinished)


def test_stop_decision_launches_nothing(campaign):
    slurm = FakeSlurm()

    def stopping_agent(prompt: str, *, cwd: Path, log_path: Path):
        (cwd.parent / "prescription.md").write_text("nothing left to try", encoding="utf-8")
        (cwd.parent / "STOP").write_text("recovery is at ceiling", encoding="utf-8")
        return True, "stopping"

    result = run_iteration(
        campaign, 1, run_agent=stopping_agent, submit_sweep=slurm.submit_sweep,
        submit_review=slurm.submit_review, prompt_template=PROMPT_TEMPLATE,
    )
    assert result.decision == "stop"
    assert slurm.sweeps == [] and slurm.reviews == []
    assert "recovery is at ceiling" in (campaign.root / "journal.md").read_text(encoding="utf-8")


def test_missing_deliverables_are_repaired_once_with_the_problems_injected(campaign):
    slurm = FakeSlurm()
    prompts: list[str] = []
    good = _good_agent(prompts)

    def forgetful_then_good(prompt: str, *, cwd: Path, log_path: Path):
        if len(prompts) == 0:
            prompts.append(prompt)
            (cwd.parent / "prescription.md").write_text("half done", encoding="utf-8")
            return True, "forgot the rest"
        return good(prompt, cwd=cwd, log_path=log_path)

    result = run_iteration(
        campaign, 1, run_agent=forgetful_then_good, submit_sweep=slurm.submit_sweep,
        submit_review=slurm.submit_review, prompt_template=PROMPT_TEMPLATE,
    )
    assert result.repairs_used == 1
    assert "next_run.env" in prompts[1] and "STOP" in prompts[1]
    assert len(slurm.sweeps) == 1


def test_persistently_missing_deliverables_fail_loudly_without_launching(campaign):
    slurm = FakeSlurm()

    def lazy_agent(prompt: str, *, cwd: Path, log_path: Path):
        (cwd.parent / "prescription.md").write_text("thoughts", encoding="utf-8")
        return True, "no decision"

    with pytest.raises(RuntimeError, match="next_run.env"):
        run_iteration(
            campaign, 1, run_agent=lazy_agent, submit_sweep=slurm.submit_sweep,
            submit_review=slurm.submit_review, prompt_template=PROMPT_TEMPLATE,
        )
    assert slurm.sweeps == [] and slurm.reviews == []


def test_uncommitted_changes_are_a_deliverable_problem(campaign):
    slurm = FakeSlurm()

    def sloppy_agent(prompt: str, *, cwd: Path, log_path: Path):
        (cwd / "src" / "loop.py").write_text("THRESHOLD = 0.9\n", encoding="utf-8")
        (cwd.parent / "prescription.md").write_text("x", encoding="utf-8")
        (cwd.parent / "next_run.env").write_text("NOTE=x\n", encoding="utf-8")
        return True, "did not commit"

    with pytest.raises(RuntimeError, match="uncommitted"):
        run_iteration(
            campaign, 1, run_agent=sloppy_agent, submit_sweep=slurm.submit_sweep,
            submit_review=slurm.submit_review, prompt_template=PROMPT_TEMPLATE,
        )
