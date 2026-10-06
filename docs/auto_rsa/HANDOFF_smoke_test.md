# Handoff: RSA loop smoke test with real Gemini agents (cloud session)

**Goal.** Find out whether Gemini agents, given the memo primer, can write memo
models that pass the RSA loop's admission gates. One round, three slots, on
the full pragmods data. This is the main risk before scaling to Sherlock.

**Where.** A fresh claude.ai cloud session on branch `auto-rsa` of
`mcfrank/auto-psych`. The environment's `GOOGLE_API_KEY` must be visible in
the session (`[ -n "$GOOGLE_API_KEY" ] && echo set`). The container is
disposable, so agents run unsandboxed (`--no-sandbox`); never do this on a
machine with other credentials.

## Steps

```bash
git checkout auto-rsa && git pull
uv sync                                  # Python 3.12, memo, jax 0.7, numpyro
npm i -g opencode-ai                     # the default agent CLI (Gemini)
export GOOGLE_GENERATIVE_AI_API_KEY="$GOOGLE_API_KEY"   # the name opencode reads
opencode --version
# 1. Can opencode reach Gemini from here? (A failure here is a network-policy
#    question, not a loop bug; see the environment's Network access settings.)
cd /tmp && opencode run -m google/gemini-3.1-pro-preview "Reply with the word ok." ; cd -
# 2. The smoke run (about 20-40 min: 3 agents, then fits of each candidate):
uv run python -m src.rsa.loop.run --results data/rsa/loop_smoke \
    --max-iterations 1 --candidate-count 3 --no-sandbox \
    --num-warmup 500 --num-samples 500 --num-chains 2
```

To try a newer model, add `--agent-model google/<model id>` (e.g. a Gemini
Flash release), and keep everything else the same so runs compare.

## What to look at

- `data/rsa/loop_smoke/report.html`: the timeline (admitted, rejected with
  reason, no file) and the standing after the round.
- `data/rsa/loop_smoke/attempted_hypotheses.jsonl`: every slot's outcome.
- `data/rsa/loop_smoke/round_1/candidate_*/agent.jsonl`: what each agent
  did. Did it run `check_candidate`? Which memo errors did it hit, and did
  the repair fix them?
- `token_usage.jsonl` (written by the launcher), for cost per slot.

## What counts as success

At least one agent-written candidate is admitted (passes the code gate,
contract, fit, convergence and novelty), and rejections carry reasons an
agent can act on. Record for each slot: admitted or why not; whether the
repair worked; tokens; wall time. Add any new memo gotchas to the primer
(`src/rsa/loop/prompts/rsa_theory.md`) and to `docs/auto_rsa/PLAN.md`.

## Then

If it works, the next handoff is a local session that runs the loop on
Sherlock with sandboxed agents and the production settings (5 rounds x 6
slots, 4 x 1000 NUTS), using `src/rsa/loop/run.py` with `--agent-root` set
to the scrubbed agent tree, as the subjective-randomness jobs do.
