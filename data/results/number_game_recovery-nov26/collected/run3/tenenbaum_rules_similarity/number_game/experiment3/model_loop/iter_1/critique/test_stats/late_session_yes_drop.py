# name: late_session_yes_drop
# description: Per-participant yes-rate change from the first to the second half of the session (trial_index >=225 minus <225), averaged; observed below null means people grow more conservative over the session than stimulus differences predict.
def test_statistic(df):
    pid = (df["trial_index"].values == 0).cumsum()
    late = df["trial_index"].values >= 225
    g = pd.DataFrame({"p": pid, "l": late, "r": df["rating"].values}).groupby(["p", "l"])["r"].mean().unstack()
    return float((g[True] - g[False]).mean())
