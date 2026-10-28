# name: repeat_exposure_decline
# description: Within each participant x set (participants start where trial_index resets), yes rate on that set's 4th-and-later presentations minus its first 3, averaged over participant-sets having both; observed below the null means people grow more conservative with a concept the more they see the same set (concept tightening/consolidation) beyond what the stationary model and target mix produce, above means they grow more liberal.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.concatenate([[0], np.cumsum(ti[1:] <= ti[:-1])])
    d = pd.DataFrame({"pid": pid, "s": df["number_set"].to_numpy(),
                      "r": df["rating"].to_numpy().astype(float)})
    d["rep"] = d.groupby(["pid", "s"]).cumcount()
    d["late"] = d["rep"] >= 3
    m = d.groupby(["pid", "s", "late"])["r"].mean().unstack()
    if True not in m.columns or False not in m.columns:
        return 0.0
    diff = (m[True] - m[False]).dropna()
    return float(diff.mean()) if len(diff) else 0.0
