# name: sequential_yes_dependence
# description: P(yes | previous trial yes) minus P(yes | previous trial no), over consecutive trials (trial_index increments by 1 in row order); observed above null means response perseveration/autocorrelation the trial-independent model lacks.
def test_statistic(df):
    r = df["rating"].astype(float).to_numpy()
    t = df["trial_index"].astype(int).to_numpy()
    if len(r) < 3:
        return 0.0
    ok = (t[1:] - t[:-1]) == 1
    prev, cur = r[:-1][ok], r[1:][ok]
    a, b = cur[prev == 1], cur[prev == 0]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
