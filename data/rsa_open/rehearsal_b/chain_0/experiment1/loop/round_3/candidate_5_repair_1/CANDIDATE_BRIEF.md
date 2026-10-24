# This slot: refine the incumbent

Improve the current best model, **bounded_depth_listener** (best on exp1 (held out, grouped CV); 195.0 ± 61.0 behind on E), with ONE stated change to its mechanism (a different functional form, prior, depth, or one component taken from another model). Your hypothesis must say what you changed and why it should fit better, and list EVERY difference between your candidate.py and its source: recursion depth (which L<k> listener calls), each parameter added or removed, and every other term. A hypothesis that misdescribes its code misleads every later round.

Its hypothesis: Listeners interpret referring expressions with a strictly bounded reasoning depth, inverting a communicative speaker who reasons about literal semantics rather than engaging in higher-order recursive mental simulation. This bounded depth prevents over-implicature, naturally preserving non-zero probability for multi-feature scalar competitors while strongly favoring minimal referents when words are universally shared. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

Its source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/bounded_depth_listener.py`

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
