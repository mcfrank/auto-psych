# This slot: refine the incumbent

Improve the current best model, **contextual_contrast_listener** (best on exp1 (held out, grouped CV); 110.2 ± 60.4 behind on E), with ONE stated change to its mechanism (a different functional form, prior, depth, or one component taken from another model). Your hypothesis must say what you changed and why it should fit better, and list EVERY difference between your candidate.py and its source: recursion depth (which L<k> listener calls), each parameter added or removed, and every other term. A hypothesis that misdescribes its code misleads every later round.

Its hypothesis: Speakers choose referring expressions to maximize contextual contrast against visually similar competitors in the immediate display, rather than simulating recursive mental states. Listeners interpret utterances by inverting this contrastive speaker, expecting heard words to discriminate the target from its nearest context competitors. On uninformative displays without a referring expression, visual singleton salience guides referent expectations.

Its source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/contextual_contrast_listener.py`

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
