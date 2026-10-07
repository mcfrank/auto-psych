# Models in the set

## literal_listener (554.5 ± 28.8 nats behind the best)

Listeners pick uniformly among the objects the word is literally true of.

Source: `data/rsa/loop_smoke2_repair/models/literal_listener.py`

## rsa_l1 (32.4 ± 7.5 nats behind the best)

Listeners invert a softmax-rational speaker who simulates a literal listener (depth 1), with a uniform prior over objects.

Source: `data/rsa/loop_smoke2_repair/models/rsa_l1.py`

## rsa_l2 (18.9 ± 7.9 nats behind the best)

Listeners reason one level deeper (depth 2), inverting a speaker who simulates the depth-1 pragmatic listener.

Source: `data/rsa/loop_smoke2_repair/models/rsa_l2.py`

## rsa_l1_salience (best)

Depth-1 RSA whose listener has a salience prior over objects (feature count, familiarization base rate) that the simulated literal listener lacks.

Source: `data/rsa/loop_smoke2_repair/models/rsa_l1_salience.py`

## rsa_l1_shared_prior (6.5 ± 1.8 nats behind the best)

Depth-1 RSA with the salience prior as common knowledge, used by the simulated literal listener as well as the pragmatic listener.

Source: `data/rsa/loop_smoke2_repair/models/rsa_l1_shared_prior.py`

## valence_evaluative_listener (admitted this round, not yet scored)

Listeners interpret the speaker as describing an object chosen according to their evaluative attitude, where desirability increases with feature complexity. When a speaker describes their favorite object, listeners infer that they favor referents with more features, whereas when describing their least favorite object, they favor referents with fewer features. Listeners invert this value-conditioned speaker upon hearing an utterance, and default to the speaker's evaluative preference when no informative word is given.

Source: `data/rsa/loop_smoke2_repair/models/valence_evaluative_listener.py`
