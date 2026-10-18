# name: ruleless_multi_far_unmatched_yes
# description: In sets of >=2 examples with no explicit rule (mixed final digits, common factor <3), yes rate for non-example targets sharing no final digit with any example and lying >=2 decades outside the examples' band; t_observed above null_mean means people give rule-less sets a broad "anything goes" acceptance the model lacks; below means the model over-generalizes far from rule-less sets.
import math
from functools import reduce
def test_statistic(df):
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        xs = [int(v) for v in s.split(",")]
        g = reduce(math.gcd, xs)
        uni = len({x % 10 for x in xs}) == 1
        ruleless = len(xs) >= 2 and not (uni or g >= 3)
        info[s] = (ruleless, min(x // 10 for x in xs), max(x // 10 for x in xs))
    rl = sets.map(lambda s: info[s][0]).to_numpy().astype(bool)
    lo = sets.map(lambda s: info[s][1]).to_numpy().astype(int)
    hi = sets.map(lambda s: info[s][2]).to_numpy().astype(int)
    t = df["target"].astype(int).to_numpy()
    ex = sets.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    in_set = (ex == t[:, None]).any(axis=1)
    digit_match = (np.mod(ex, 10) == (t % 10)[:, None]).any(axis=1)
    td = t // 10
    gap = np.maximum(np.maximum(lo - td, 0), td - hi)
    m = rl & ~in_set & ~digit_match & (gap >= 2)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).to_numpy()[m].mean())
