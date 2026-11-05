# name: multi_above_below_asymmetry
# description: On 2-4-example sets, yes rate on targets above the largest example minus yes rate on targets below the smallest; observed > null means people extend upward more readily than downward, an asymmetry the model's equal-width band under-produces (observed < null: model over-produces it).
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1).values
    lo = ex.min(axis=1).values
    hi = ex.max(axis=1).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    above = (n >= 2) & (t > hi)
    below = (n >= 2) & (t < lo)
    if above.sum() == 0 or below.sum() == 0:
        return 0.0
    return float(r[above].mean() - r[below].mean())
