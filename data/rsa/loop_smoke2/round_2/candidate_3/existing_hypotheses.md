# Models in the set

## literal_listener (1559.6 ± 51.2 nats behind the best)

Listeners pick uniformly among the objects the word is literally true of.

Source: `data/rsa/loop_smoke2/models/literal_listener.py`

## rsa_l1 (129.0 ± 16.6 nats behind the best)

Listeners invert a softmax-rational speaker who simulates a literal listener (depth 1), with a uniform prior over objects.

Source: `data/rsa/loop_smoke2/models/rsa_l1.py`

## rsa_l2 (81.2 ± 16.6 nats behind the best)

Listeners reason one level deeper (depth 2), inverting a speaker who simulates the depth-1 pragmatic listener.

Source: `data/rsa/loop_smoke2/models/rsa_l2.py`

## rsa_l1_salience (94.2 ± 14.2 nats behind the best)

Depth-1 RSA whose listener has a salience prior over objects (feature count, familiarization base rate) that the simulated literal listener lacks.

Source: `data/rsa/loop_smoke2/models/rsa_l1_salience.py`

## rsa_l1_shared_prior (90.3 ± 14.1 nats behind the best)

Depth-1 RSA with the salience prior as common knowledge, used by the simulated literal listener as well as the pragmatic listener.

Source: `data/rsa/loop_smoke2/models/rsa_l1_shared_prior.py`

## valence_salience_listener (37.5 ± 9.0 nats behind the best)

Listeners interpret the speaker's evaluative framing (favorite versus least favorite) as modulating expectations of referent complexity: people expect a speaker with positive valence to describe a feature-rich object, but expect a speaker with negative valence to describe a simple, feature-poor object. This valence-directed prior over referent richness guides both prior expectations and pragmatic inversion of the speaker's utterance.

Source: `data/rsa/loop_smoke2/models/valence_salience_listener.py`

## rsa_l2_shared_prior (52.4 ± 14.6 nats behind the best)

We refine depth-2 RSA (rsa_l2) by incorporating the shared salience prior from rsa_l1_shared_prior as common knowledge across all levels of recursive reasoning. Rather than assuming uniform object priors, speakers and listeners at every depth weight referents by their familiarization base rates and feature counts, allowing prior object expectations to guide pragmatic disambiguation as well as uninformative prior choices.

Source: `data/rsa/loop_smoke2/models/rsa_l2_shared_prior.py`

## rsa_l2_shared_prior_2 (best)

We refine rsa_l1_shared_prior by extending its recursive pragmatic reasoning depth from depth 1 to depth 2, so that listeners invert a depth-2 speaker who simulates the depth-1 pragmatic listener rather than a literal listener. Under this mechanism, the salience prior (grounded in feature counts and familiarization base rates) remains common knowledge across all levels of recursion, allowing listeners to resolve complex implicatures while respecting prior salience.

Source: `data/rsa/loop_smoke2/models/rsa_l2_shared_prior_2.py`
