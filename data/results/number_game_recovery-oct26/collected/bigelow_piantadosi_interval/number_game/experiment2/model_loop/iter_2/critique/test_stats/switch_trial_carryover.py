# name: switch_trial_carryover
# description: Among consecutive within-participant trial pairs where the example set CHANGES, yes rate after a yes minus yes rate after a no; observed above the null means response-level perseveration that carries across different sets (not explained by repeated-set stimulus similarity), which the trial-independent model under-produces.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy().astype(float)
    s = df["number_set"].to_numpy()
    ok = (ti[1:] > ti[:-1]) & (s[1:] != s[:-1])
    prev, cur = r[:-1][ok], r[1:][ok]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
