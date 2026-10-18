# name: rule_set_obey_minus_violate_gap
# description: In sets of >=2 examples with an explicit rule (all share a final digit, or common factor g>=3), yes rate for non-example targets obeying the rule (same final digit / multiple of g) minus targets violating it; t_observed above null_mean means people apply detected rules more sharply (all-or-none) than the model's graded features; below means the model over-sharpens rules.
import math
from functools import reduce
def test_statistic(df):
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        xs = [int(v) for v in s.split(",")]
        if len(xs) < 2:
            info[s] = (0, -1, 0); continue
        digs = {x % 10 for x in xs}
        g = reduce(math.gcd, xs)
        if len(digs) == 1:
            info[s] = (1, xs[0] % 10, 0)
        elif g >= 3:
            info[s] = (2, -1, g)
        else:
            info[s] = (0, -1, 0)
    kind = sets.map(lambda s: info[s][0]).to_numpy()
    dig = sets.map(lambda s: info[s][1]).to_numpy()
    g = sets.map(lambda s: info[s][2]).to_numpy()
    t = df["target"].astype(int).to_numpy()
    ex = sets.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    in_set = (ex == t[:, None]).any(axis=1)
    gsafe = np.where(g > 0, g, 1)
    obey = np.where(kind == 1, (t % 10) == dig, np.where(kind == 2, (t % gsafe) == 0, False))
    rule = (kind > 0) & ~in_set
    y = df["rating"].astype(float).to_numpy()
    a = y[rule & obey]; b = y[rule & ~obey]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
