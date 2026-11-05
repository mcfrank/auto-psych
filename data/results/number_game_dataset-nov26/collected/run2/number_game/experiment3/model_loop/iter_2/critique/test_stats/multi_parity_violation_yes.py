# name: multi_parity_violation_yes
# description: On 2-4-example sets whose examples all share parity, yes rate on targets of the opposite parity; observed > null means people accept parity-violating targets more than the model (its shared-profile share is too strict), observed < null the model is too lenient.
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1).values
    par = ex % 2
    mx = par.max(axis=1).values
    mn = par.min(axis=1).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    sel = (n >= 2) & (mx == mn) & ((t % 2) != mn)
    if sel.sum() == 0:
        return 0.0
    return float(r[sel].mean())
