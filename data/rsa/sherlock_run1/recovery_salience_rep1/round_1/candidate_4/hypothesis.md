Listeners reason at depth 2 about a pragmatic speaker, but rather than simply probability matching their posterior beliefs, they apply softmax decision rationality when selecting an object to click. This separates speaker communicative optimization from listener choice decisiveness, allowing the model to capture that listeners decisively select the referent with the highest posterior support even when speaker informativeness is moderate.

Model refinement details:
- Base model refined: rsa_l2
- Recursion depth: Depth 2 (choice_probs calls L2)
- Parameters added: beta (LogNormal(0.0, 1.0)), representing listener decision rationality
- Parameters removed: None
- Structural differences: In L2, the listener's choice weights are exponentiated by beta times the log posterior probability of each referent given the utterance, rather than matching posterior probability directly (beta = 1).
