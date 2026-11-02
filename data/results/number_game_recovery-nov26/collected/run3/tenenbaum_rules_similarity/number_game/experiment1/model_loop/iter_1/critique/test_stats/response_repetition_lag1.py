# name: response_repetition_lag1
# description: Within-participant (blocks from trial_index resets) P(yes | previous trial yes) minus P(yes | previous trial no); observed > model means sequential response perseveration/inertia beyond what stimulus-driven model probabilities produce.
def test_statistic(df):
    t = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy().astype(float)
    if len(r) < 3:
        return 0.0
    valid = np.diff(t) > 0
    prev, cur = r[:-1][valid], r[1:][valid]
    a, b = prev == 1, prev == 0
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(cur[a].mean() - cur[b].mean())
