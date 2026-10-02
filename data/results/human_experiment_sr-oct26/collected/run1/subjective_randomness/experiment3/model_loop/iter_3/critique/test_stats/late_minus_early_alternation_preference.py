# name: late_minus_early_alternation_preference
# description: Among pairs whose switch counts differ, rate of choosing the more-alternating sequence in the second half of a session minus the first half; observed above null_mean means people drift toward favouring alternation over the session (the model's judgement is stationary), below means they drift away.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    def sw_overlap(s):
        x = s.to_numpy().astype(str)
        L = s.str.len().to_numpy()
        m = max(L.max(), 2)
        arr = np.array([list(v.ljust(m, " ")) for v in x])
        return ((arr[:, 1:] != arr[:, :-1]) & (arr[:, 1:] != " ") & (arr[:, :-1] != " ")).sum(1)
    sa = sw_overlap(a); sb = sw_overlap(b)
    diff = sa - sb
    mask = diff != 0
    chose_more = np.where(diff > 0, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy()).astype(float)
    late = df["trial_index"].to_numpy() > 32
    e = chose_more[mask & ~late]; l = chose_more[mask & late]
    if len(e) == 0 or len(l) == 0:
        return 0.0
    return float(l.mean() - e.mean())
