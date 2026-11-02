# name: same_set_response_perseveration
# description: Within a participant (trial_index resets) and on consecutive trials with the same number_set, P(yes | previous answer yes) minus P(yes | previous answer no); observed above null_mean means answers carry over trial to trial (perseveration) beyond what the stimulus-driven model produces.
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy()
    pid = np.cumsum(np.r_[True, ti[1:] <= ti[:-1]])
    ns = df["number_set"].to_numpy()
    r = df["rating"].astype(float).to_numpy()
    ok = (pid[1:] == pid[:-1]) & (ns[1:] == ns[:-1])
    prev = r[:-1][ok]; cur = r[1:][ok]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
