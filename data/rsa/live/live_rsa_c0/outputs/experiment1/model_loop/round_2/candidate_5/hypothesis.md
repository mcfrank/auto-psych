RSA pragmatic listener at depth 2 with utterance specificity costs, competitor confusion avoidance, singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0. Refining rsa_l2_confusion_singleton_l0 by incorporating the utterance specificity cost component from specificity_cost_speaker: speakers in reference games experience cognitive specificity and verification costs when articulating uniquely distinguishing expressions, leading them to frequently favor broader, lower-cost shared referring expressions even when describing feature-rich objects. Pragmatic listeners invert this cost-sensitive, confusion-averse speaker at depth 2, recognizing that hearing a shared or ubiquitous word does not rule out complex referents whose unique features carry higher production costs, which attenuates overly aggressive scalar implicatures in two-object displays and increases feature-rich referent selections under ubiquitous expressions.

Differences from rsa_l2_confusion_singleton_l0:
- Recursion depth remains depth 2 (choice_probs calls L2).
- Added parameter c_spec (prior dist.Normal(0.0, 1.0)).
- Added utterance specificity cost penalty term - c_spec * vec(cost, u) to speaker utility in both S1 and S2.
- Added computation of utterance specificity (cost = (1.0 - extension) * (1.0 - ctx.is_sink)) in choice_probs and passed cost to L1 and L2.
- No parameters removed.
