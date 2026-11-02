# name: cross_set_carryover_effect
# description: Within participant, yes rate when the immediately preceding trial was a yes on a DIFFERENT set minus when it was a no on a different set; the model treats trials independently (null near its stimulus-driven baseline), so t_observed above null_mean means response perseveration/carryover across sets.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    s = df["number_set"].astype(str).to_numpy()
    r = df["rating"].to_numpy(dtype=float)
    ok = np.r_[False, (np.diff(ti) == 1) & (s[1:] != s[:-1])]
    prev = np.r_[np.nan, r[:-1]]
    a = ok & (prev == 1)
    b = ok & (prev == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
