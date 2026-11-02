# name: shared_units_digit_nonmatch_leakage
# description: Yes rate for targets whose units digit differs from the examples' common units digit, among multi-example sets whose examples all end in the same digit; observed above null means people hedge beyond an "ends in d" reading (via ranges/similarity) more than the model predicts, i.e. the model commits too hard to the units-digit rule; below means people commit even more.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    nums = parts.apply(pd.to_numeric, errors="coerce")
    units = nums % 10
    n = nums.notna().sum(axis=1)
    umin = units.min(axis=1); umax = units.max(axis=1)
    t = df["target"].astype(int)
    m = (n >= 2) & (umin == umax) & ((t % 10) != umin)
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].astype(float).mean())
