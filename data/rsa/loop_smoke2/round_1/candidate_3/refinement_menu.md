# Refinement menu

Choose ONE model to refine (not the incumbent). Your hypothesis must name it and state the single change you make. Live models first, then models pruned earlier (a pruned mechanism may come back with a substantive change).

## literal_listener (1478.4 ± 49.4 nats behind the best)

Listeners pick uniformly among the objects the word is literally true of.

Source: `data/rsa/loop_smoke2/models/literal_listener.py`

## rsa_l1 (47.8 ± 3.9 nats behind the best)

Listeners invert a softmax-rational speaker who simulates a literal listener (depth 1), with a uniform prior over objects.

Source: `data/rsa/loop_smoke2/models/rsa_l1.py`

## rsa_l1_salience (13.0 ± 8.4 nats behind the best)

Depth-1 RSA whose listener has a salience prior over objects (feature count, familiarization base rate) that the simulated literal listener lacks.

Source: `data/rsa/loop_smoke2/models/rsa_l1_salience.py`

## rsa_l1_shared_prior (9.1 ± 8.5 nats behind the best)

Depth-1 RSA with the salience prior as common knowledge, used by the simulated literal listener as well as the pragmatic listener.

Source: `data/rsa/loop_smoke2/models/rsa_l1_shared_prior.py`
