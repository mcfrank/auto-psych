# Refinement menu

Choose ONE model to refine (not the incumbent). Your hypothesis must name it and state the single change you make. Live models first, then models pruned earlier (a pruned mechanism may come back with a substantive change).

## literal_listener (554.5 ± 28.8 nats behind the best)

Listeners pick uniformly among the objects the word is literally true of.

Source: `data/rsa/loop_smoke/models/literal_listener.py`

## rsa_l1 (32.4 ± 7.5 nats behind the best)

Listeners invert a softmax-rational speaker who simulates a literal listener (depth 1), with a uniform prior over objects.

Source: `data/rsa/loop_smoke/models/rsa_l1.py`

## rsa_l2 (18.9 ± 7.9 nats behind the best)

Listeners reason one level deeper (depth 2), inverting a speaker who simulates the depth-1 pragmatic listener.

Source: `data/rsa/loop_smoke/models/rsa_l2.py`

## rsa_l1_shared_prior (6.5 ± 1.8 nats behind the best)

Depth-1 RSA with the salience prior as common knowledge, used by the simulated literal listener as well as the pragmatic listener.

Source: `data/rsa/loop_smoke/models/rsa_l1_shared_prior.py`
