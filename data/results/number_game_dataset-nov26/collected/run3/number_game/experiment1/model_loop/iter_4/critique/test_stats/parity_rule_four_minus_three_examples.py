# name: parity_rule_four_minus_three_examples
# description: For sets whose examples all share parity but not a last digit, yes rate for same-parity non-example targets in 4-example sets minus 3-example sets; the size principle predicts a positive gap (more examples, stronger parity rule), so observed below null means extra examples do not strengthen (or even weaken) parity generalization as the model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    par = ex % 2; last = ex % 10
    uni_par = (par.max(axis=1) == par.min(axis=1))
    uni_last = (last.max(axis=1) == last.min(axis=1))
    inset = ex.eq(t, axis=0).any(axis=1)
    same = (t % 2) == par.iloc[:, 0]
    base = uni_par & ~uni_last & ~inset & same
    a = base & (n == 4); b = base & (n == 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
