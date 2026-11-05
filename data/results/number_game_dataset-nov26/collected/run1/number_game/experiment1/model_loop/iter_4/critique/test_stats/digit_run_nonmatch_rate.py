# name: digit_run_nonmatch_rate
# description: Yes-rate, over all trials of sets with >=3 examples that all share one final digit, for targets that do NOT have that final digit; observed < null means people treat a run of same-final-digit examples as a strict rule and reject non-matching targets more than the model predicts, observed > null means more lenient than predicted.
def test_statistic(df):
    sets = df["number_set"].unique()
    dig = {}
    for s in sets:
        ex = [int(x) for x in s.split(",")]
        if len(ex) >= 3 and len(set(e % 10 for e in ex)) == 1:
            dig[s] = ex[0] % 10
    d = df["number_set"].map(dig)
    mask = d.notna() & (df["target"].astype(int) % 10 != d.fillna(-1))
    if mask.sum() == 0:
        return 0.0
    return float(df.loc[mask, "rating"].mean())
