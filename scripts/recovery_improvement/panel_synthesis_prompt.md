# Review panel `$panel_name` — synthesis (you are the moderator, **$moderator**)

A panel of research engineers has reviewed two automated cognitive-model discovery loops over $total_rounds written rounds (independent reviews, then discussion). You have read nothing yet; the whole thread is below. Your job is to turn it into one plan the user can act on.

Members:
$members_list

Repos (read-only): auto-psych at `$repo`, llm-verbal-protocol at `$verbal_repo`. Python: `$venv_py`. Scratch for anything you compute: `$panel_root/scratch/synthesis/` (create it).

$capabilities
If you cannot write files, your final message is the plan; put the sweep spec, if any, in a fenced block that starts with ```next_run.env and the wrapper will write it next to the plan.

## What to produce

Write `$plan_path` (Markdown) with:

1. **Consensus** — the findings every member accepted, each with the evidence the thread settled on.
2. **Plan for the auto-psych loop** — ranked items. For each: the change (mechanism + file), why (the evidence), expected effect, the observation in the next recovery sweep that confirms or refutes it, risk, and who proposed / who dissented. Item 1 is what gets implemented first, so it must be one coherent change.
3. **Plan for the verbal-protocol loop** — the same, for that repo.
4. **What each loop should borrow from the other** — concrete, with the file on each side.
5. **Unresolved disagreements** — stated fairly, with what evidence would settle each.
6. **Rejected proposals** — and why.

Then, if item 1 of the auto-psych plan is worth testing with a recovery sweep, write `$next_run_path`: `KEY=value` lines overriding the campaign's sweep defaults (an empty file = the defaults), plus a `NOTE=` line saying what the sweep tests. Allowed keys:
$allowed_keys

Campaign defaults:
```
$sweep_defaults
```
Write no `next_run.env` if the plan's first item needs no sweep (say so in the plan).

## Rules

- Be a moderator, not a sixth reviewer: weigh the evidence the members brought, resolve disagreements on that evidence, and go back to the files only to check a contested claim. Say when a member's claim did not survive checking.
- Keep every recommendation general — never a hint about which model is held out in either benchmark.
- Do not modify either repository, launch or cancel jobs, or `git push`.

$repair_feedback

## The thread

$thread
