"""Autonomous recovery-improvement campaign.

A campaign is a chain of Slurm jobs in which a Claude Code review agent reads
the most recent holdout-recovery sweeps, writes a prescription for improving
the model-recovery loop, commits its tweaks on a per-iteration branch of the
subject repo, and declares the sweep that should test them. Deterministic
Python (this package) does the bookkeeping around the agent: digesting sweep
results, validating the agent's deliverables, launching the sweep, and chaining
the next review job. All state crosses iterations on disk, under the campaign
root on ``$SCRATCH``.
"""
