# name: rule_set_in_band_nonrule_yes
# description: In sets of >=2 examples with an explicit rule (shared final digit or common factor >=3), yes rate for non-example targets that violate the rule, share no final digit with any example, but lie within the examples' decade band; t_observed above null_mean means magnitude familiarity survives rule detection more than the model's magnitude_preempt allows; below means rules suppress "nearby" more.
import math
from functools import reduce
def test_statistic(df):
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        xs = [int(v) for v in s.split(",")]
        g = reduce(math.gcd, xs)
        uni = len({x % 10 for x in xs}) == 1
        rule = len(xs) >= 2 and (uni or g >= 3)
        info[s] = (rule, g if g >= 3 else 0, min(x // 10 for x in xs), max(x // 10 for x in xs))
    rule = sets.map(lambda s: info[s][0]).to_numpy().astype(bool)
    g = sets.map(lambda s: info[s][1]).to_numpy().astype(int)
    lo = sets.map(lambda s: info[s][2]).to_numpy().astype(int)
    hi = sets.map(lambda s: info[s][3]).to_numpy().astype(int)
    t = df["target"].astype(int).to_numpy()
    ex = sets.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    in_set = (ex == t[:, None]).any(axis=1)
    digit_match = (np.mod(ex, 10) == (t % 10)[:, None]).any(axis=1)
    mult = (g > 0) & (t % np.where(g > 0, g, 1) == 0)
    td = t // 10
    in_band = (td >= lo) & (td <= hi)
    m = rule & ~in_set & ~digit_match & ~mult & in_band
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).to_numpy()[m].mean())
