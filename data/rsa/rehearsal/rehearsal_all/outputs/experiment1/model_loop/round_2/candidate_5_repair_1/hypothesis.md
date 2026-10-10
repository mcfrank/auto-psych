Pragmatic listeners reason at depth two about a speaker who balances competitor-confusion aversion with competitor-reliance sensitivity. When communicating, speakers actively avoid referring expressions that apply to visually confusable competitors or that deprive alternative referents of their primary identifying labels, choosing words that minimize referential interference. Pragmatic listeners invert this confusion- and reliance-sensitive speaker while integrating a shared perceptual prior favoring visual singletons, distinctive objects, feature complexity under positive framing, and high color contrast.

Refining rsa_l2_competitor_confusion_l0 by incorporating competitor-reliance sensitivity from competitor_reliance_speaker into the speaker utility. This should fit better by capturing participants' sensitivity to competitor descriptive options (explaining referent choices in contexts like Mayn & Demberg 2023 where competitor reliance excels) while maintaining depth-two recursive reasoning, competitor-confusion aversion, and perceptual salience priors.

Differences from source (rsa_l2_competitor_confusion_l0):
- Recursion depth: Unchanged; choice_probs calls depth-2 listener L2.
- Parameters added: w_reliance ~ Normal(0.0, 2.0).
- Parameters removed: None.
- Terms changed: S1 and S2 speaker choice utilities additionally subtract w_reliance * at(reliance, u, r), where reliance is the competitor reliance matrix measuring the sum of competitor inverse feature counts for that word. The competitor confusion calculation, perceptual prior computation, literal L0 semantics, depth-2 recursive reasoning, and lapse process remain identical to the source.
