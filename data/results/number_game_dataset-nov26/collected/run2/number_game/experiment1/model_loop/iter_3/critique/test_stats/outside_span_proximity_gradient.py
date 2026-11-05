# name: outside_span_proximity_gradient
# description: Among targets outside the examples' [min,max] range (multi-example sets), yes-rate for targets within 10 of the range minus those more than 10 away; the model treats all out-of-span targets alike, so observed > null means people show a graded numeric-proximity generalization the model lacks (observed < null: the reverse).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    lo = ex.map(lambda l: min(int(x) for x in l)).values
    hi = ex.map(lambda l: max(int(x) for x in l)).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0))
    out = (dist > 0) & (n >= 2)
    near = out & (dist <= 10)
    far = out & (dist > 10)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
