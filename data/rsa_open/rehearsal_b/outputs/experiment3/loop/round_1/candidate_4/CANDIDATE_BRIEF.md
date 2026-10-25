# This slot: refine the incumbent

Improve the current best model, **feature_uncertainty_listener_2** (best on exp3 (held out, grouped CV); 139.5 ± 56.5 behind on E, 36.5 ± 18.7 behind on exp1, 44.4 ± 42.6 behind on exp2), with ONE stated change to its mechanism (a different functional form, prior, depth, or one component taken from another model). Your hypothesis must say what you changed and why it should fit better, and list EVERY difference between your candidate.py and its source: recursion depth (which L<k> listener calls), each parameter added or removed, and every other term. A hypothesis that misdescribes its code misleads every later round.

Its hypothesis: Listeners interpret referring expressions under referential feature uncertainty, being unsure which visual feature of a candidate referent the speaker intended to pick out. When evaluating literal semantics, listeners weight an object's compatibility by the probability that the speaker chose the heard feature from that object's feature bundle, naturally favoring minimal referents whose features are exhaustively described over cluttered multi-feature competitors. Communicative speakers select expressions expecting this feature-attribution inference, while spontaneous visual attention on uninformative displays defaults to contextually unique singletons.

Its source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment3/loop/models/feature_uncertainty_listener_2.py`

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
