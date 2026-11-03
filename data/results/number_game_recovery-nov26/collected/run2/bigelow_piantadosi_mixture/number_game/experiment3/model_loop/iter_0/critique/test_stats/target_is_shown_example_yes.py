# name: target_is_shown_example_yes
# description: Yes rate when the target is one of the shown examples; observed below null means people deny shown examples (lapse/no-repeat) more often than the model predicts, above means less.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mx = x.max(axis=1)
    mn = x.min(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    v = r[isin].mean()
    return float(0.0 if np.isnan(v) else v)
