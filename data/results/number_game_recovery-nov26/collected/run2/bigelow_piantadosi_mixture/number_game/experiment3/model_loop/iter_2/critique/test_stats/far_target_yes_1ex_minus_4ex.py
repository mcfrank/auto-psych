# name: far_target_yes_1ex_minus_4ex
# description: Yes rate on targets more than 30 from every example for single-example sets minus for 4-example sets; observed above null means one example leaves people more permissive far away (relative to four) than the model's constant lapse plus rule mixture predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    dist = x.sub(t, axis=0).abs()
    nd = dist.min(axis=1)
    far = nd > 30
    v = r[far & (n == 1)].mean() - r[far & (n == 4)].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
