# name: same_set_lag1_response_agreement
# description: Among consecutive trials (trial_index increasing by 1) with the same number_set, fraction where the response repeats the previous response; observed above null means response perseveration/sequential dependence the independent-trial model lacks, below null means alternation.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    n = (~np.isnan(ex)).sum(1)
    mx = np.nanmax(ex, 1); mn = np.nanmin(ex, 1)
    isex = (ex == t[:, None]).any(1)
    ti = df["trial_index"].astype(int).values
    s = df["number_set"].astype(str).values
    if len(r) < 2:
        return 0.0
    same = (s[1:] == s[:-1]) & (ti[1:] == ti[:-1] + 1)
    if same.sum() == 0:
        return 0.0
    return float((r[1:] == r[:-1])[same].mean())
