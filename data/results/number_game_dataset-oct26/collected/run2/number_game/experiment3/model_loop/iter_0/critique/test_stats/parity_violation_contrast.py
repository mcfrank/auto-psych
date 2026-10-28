# name: parity_violation_contrast
# description: For sets of >=2 examples that all share one parity, yes-rate on opposite-parity targets minus same-parity targets; observed < null means people reject parity-violating targets more strongly than the model's graded parity similarity produces (observed > null: model over-penalizes parity).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    k = ex.str.len()
    odd = ex.map(lambda e: sum(int(x) % 2 for x in e))
    allsame = (k >= 2) & ((odd == 0) | (odd == k))
    setpar = (odd == k).astype(int)
    t = df["target"].astype(int)
    viol = (t % 2) != setpar
    r = df["rating"].astype(float)
    a = r[allsame & viol]
    b = r[allsame & ~viol]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
