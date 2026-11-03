# name: participant_set_breadth_heterogeneity
# description: Mean over sets (>= 15 trials) of the across-participant variance of per-participant yes rates on that set (participants = consecutive blocks starting at trial_index 0); observed > null means people differ in how broadly they generalise from a given set (different chosen concepts) beyond the model's independent-response binomial noise.
def test_statistic(df):
    pid = np.cumsum(df["trial_index"].values == 0)
    d = pd.DataFrame({"s": df["number_set"].values, "p": pid, "r": df["rating"].astype(float).values})
    cnt = d.groupby("s")["r"].size()
    keep = cnt[cnt >= 15].index
    d = d[d["s"].isin(keep)]
    if len(d) == 0:
        return 0.0
    m = d.groupby(["s", "p"])["r"].mean()
    v = m.groupby(level=0).var(ddof=0)
    return float(v.mean())
