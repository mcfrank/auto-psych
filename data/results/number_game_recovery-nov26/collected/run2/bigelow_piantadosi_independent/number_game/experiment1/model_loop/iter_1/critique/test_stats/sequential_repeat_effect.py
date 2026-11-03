# name: sequential_repeat_effect
# description: P(yes | previous trial yes) minus P(yes | previous trial no), over consecutive rows with trial_index increasing by 1 and the same number_set; observed > null means responses are more autocorrelated (perseveration) than the model's independent-trial predictions.
def test_statistic(df):
    y = df["rating"].to_numpy(dtype=float)
    ti = df["trial_index"].to_numpy()
    ns = df["number_set"].astype(str).to_numpy()
    ok = (ti[1:] == ti[:-1] + 1) & (ns[1:] == ns[:-1])
    prev = y[:-1][ok]; cur = y[1:][ok]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
