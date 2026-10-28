# name: prev_yes_carryover
# description: Over consecutive within-session trial pairs (trial_index increments by 1), P(yes | previous yes) minus P(yes | previous no); observed above the null means sequential response perseveration that the trial-independent model under-produces, below means alternation.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy()
    ok = ti[1:] == ti[:-1] + 1
    prev, cur = r[:-1][ok], r[1:][ok]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
