# name: parity_rule_distance_gradient
# description: Among 2+ example sets whose examples all share parity, for parity-matching non-example targets, yes rate within 10 of the nearest example minus yes rate 25+ away; observed > null means people's rule-consistent generalisation decays with numerical distance more than the model's flat rule extensions, observed < null means less decay.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    n = ex.notna().sum(axis=1)
    par = ex % 2
    allodd = (par.eq(1) | ex.isna()).all(axis=1)
    alleven = (par.eq(0) | ex.isna()).all(axis=1)
    match = (allodd & (t % 2 == 1)) | (alleven & (t % 2 == 0))
    inset = ex.eq(t, axis=0).any(axis=1)
    dist = ex.sub(t, axis=0).abs().min(axis=1)
    base = (n >= 2) & match & ~inset
    r = df["rating"].astype(float)
    a = r[base & (dist <= 10)]
    b = r[base & (dist >= 25)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
