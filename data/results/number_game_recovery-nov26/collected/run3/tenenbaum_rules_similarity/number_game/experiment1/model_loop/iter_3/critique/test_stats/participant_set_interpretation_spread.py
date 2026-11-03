# name: participant_set_interpretation_spread
# description: Mean, over example sets with >= 8 distinct targets, of the variance across participants (blocks delimited by trial_index resets) of each participant's yes-rate on that set; observed above model means individuals each adopt a different hypothesis for a set (broad vs narrow, hypothesis sampling) rather than all responding from one shared averaged posterior, below means people are more uniform than independent Bernoulli responding implies.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[True, np.diff(ti) < 0])
    d = pd.DataFrame({"s": df["number_set"].astype(str).to_numpy(), "t": df["target"].to_numpy(),
                      "p": pid, "r": df["rating"].to_numpy()})
    nt = d.groupby("s")["t"].nunique()
    keep = nt[nt >= 8].index
    d = d[d["s"].isin(keep)]
    if len(d) == 0:
        return 0.0
    ps = d.groupby(["s", "p"])["r"].mean().reset_index()
    v = ps.groupby("s")["r"].var(ddof=0)
    return float(v.mean())
