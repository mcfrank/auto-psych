# name: single_example_multiple_other_digit_yes
# description: Among single-example sets with example 2-12, yes rate for targets that are multiples of the example but do not share its final digit or decade and are not the example; t_observed > null_mean means the model under-produces "multiples of n" generalization from one example (its factor boost is too weak / absent for n<3).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    single = ~sets.str.contains(",")
    x = pd.to_numeric(sets.where(single, "0"), errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    ok = single.values & (x >= 2) & (x <= 12)
    xs = np.where(x > 0, x, 1)
    mult = (t % xs == 0)
    mask = ok & mult & (t != x) & (t % 10 != x % 10) & (t // 10 != x // 10)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
