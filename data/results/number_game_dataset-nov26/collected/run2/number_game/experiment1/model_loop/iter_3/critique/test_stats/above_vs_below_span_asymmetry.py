# name: above_vs_below_span_asymmetry
# description: Yes-rate for targets larger than every example minus yes-rate for targets smaller than every example (multi-example sets); the model's span term is symmetric, so observed > null means people extend upward more readily than downward (observed < null: downward more readily).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    lo = ex.map(lambda l: min(int(x) for x in l)).values
    hi = ex.map(lambda l: max(int(x) for x in l)).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    a = (t > hi) & (n >= 2)
    b = (t < lo) & (n >= 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
