# name: yes_rate_decline_over_session
# description: Yes-rate in the first third of each session (trial_index < 150) minus the last third (trial_index >= 300); observed above null_mean means people grow more conservative over the session than the time-invariant model predicts from the stimuli alone.
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    early = r[ti < 150]
    late = r[ti >= 300]
    if len(early) == 0 or len(late) == 0:
        return 0.0
    return float(early.mean() - late.mean())
