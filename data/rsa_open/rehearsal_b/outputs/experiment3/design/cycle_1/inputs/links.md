# The link library

Each link scores one response type. Every link has a decision `temperature` (sharpens or flattens the theory's probabilities) and a `lapse` (a uniform guess); some have more. Their parameters are fitted jointly with the theories', shared by every theory on the paradigm. The paradigm fixes the options (scale points, total points).

## `choice` (response type `choice`)

Pick one referent: the tempered listener posterior, mixed with a uniform guess with probability lapse.

Parameters: `lapse` ~ Beta, `temperature` ~ LogNormal.

## `rating` (response type `rating`)

Rate how likely a probed referent is: an ordered probit on the logit of its tempered probability, equally spaced cutpoints, and a uniform-rating lapse.

Parameters: `center` ~ Normal, `spread` ~ LogNormal, `lapse` ~ Beta, `temperature` ~ LogNormal.

## `allocation` (response type `bet`)

Split points over the referents (a bet): a Dirichlet-multinomial around the tempered posterior mixed with a uniform guess.

Parameters: `kappa` ~ LogNormal, `lapse` ~ Beta, `temperature` ~ LogNormal.

## `ordered_beta` (response type `slider`)

A 0-max slider on a probed referent: ordered beta, with point masses at 0 and max and a Beta in between around its tempered probability.

Parameters: `cut_low` ~ Normal, `cut_gap` ~ LogNormal, `phi` ~ LogNormal, `lapse` ~ Beta, `temperature` ~ LogNormal.

## `production` (response type `production`)

Choose a message for a target: the tempered speaker distribution over the scene's real messages, with a uniform lapse among them.

Parameters: `lapse` ~ Beta, `temperature` ~ LogNormal.

## `binary_judgment` (response type `binary_judgment`)

Would the speaker say this message for this target? A probit on the logit of the tempered speaker probability, with a bias and a random yes/no lapse.

Parameters: `center` ~ Normal, `lapse` ~ Beta, `temperature` ~ LogNormal.

The EIG tool scores every response type. Two are scored on a coarser view of the response, so their score is a lower bound: a slider binned into 12 outcomes, and a bet as a bet of at most 10 points (a random part of a bet is a smaller bet).
