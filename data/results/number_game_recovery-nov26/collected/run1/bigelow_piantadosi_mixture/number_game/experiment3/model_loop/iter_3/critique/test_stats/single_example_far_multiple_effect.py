# name: single_example_far_multiple_effect
# description: For single-example sets with example 3-30, yes rate for targets that are multiples of the example and >=15 away minus yes rate for non-multiples >=15 away; observed > null means people extend a lone example's multiples rule to far numbers more than the model (with distance hesitance and first-example rule guess) predicts, observed < null less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    single = ~sets.str.contains(",")
    x = pd.to_numeric(sets.where(single, "0"), errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    ok = single.values & (x >= 3) & (x <= 30) & (np.abs(t - x) >= 15)
    xs = np.where(x > 0, x, 1)
    mult = (t % xs) == 0
    r = df["rating"].values.astype(float)
    a, b = ok & mult, ok & ~mult
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
