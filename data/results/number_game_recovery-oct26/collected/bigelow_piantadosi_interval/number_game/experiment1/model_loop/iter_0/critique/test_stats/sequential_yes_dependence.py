# name: sequential_yes_dependence
# description: P(yes | previous trial yes) minus P(yes | previous trial no) over consecutive trials (trial_index step of 1); observed above null means response perseveration / carry-over that the trial-independent model does not produce.
def test_statistic(df):
    ti = df["trial_index"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    ok = np.diff(ti) == 1
    prev, cur = r[:-1][ok], r[1:][ok]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
