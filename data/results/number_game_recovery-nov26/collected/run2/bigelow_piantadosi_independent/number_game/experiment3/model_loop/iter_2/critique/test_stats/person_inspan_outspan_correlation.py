# name: person_inspan_outspan_correlation
# description: Across participants (trial_index resets; rates centred on their 5-person design batch), correlation between a person's yes-rate on in-span non-example targets and on out-of-span targets, in 2+-example sets; observed above null_mean means people carry a shared generalization-breadth/yes-bias trait the model lacks, below means a trade-off (range fill-in vs rule extension) stronger than the model's per-person range looseness produces.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.concatenate([[0], np.cumsum(np.diff(ti) < 0)])
    batch = pid // 5
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    isex = (ex == t[:, None]).any(axis=1)
    y = df["rating"].astype(float).to_numpy()
    ins = (n >= 2) & (t > lo) & (t < hi) & ~isex
    out = (n >= 2) & ((t < lo) | (t > hi))
    def rates(mask):
        d = pd.DataFrame({"p": pid[mask], "b": batch[mask], "y": y[mask]})
        pr = d.groupby("p").agg(y=("y", "mean"), b=("b", "first"))
        pr["y"] = pr["y"] - pr.groupby("b")["y"].transform("mean")
        return pr["y"]
    a = rates(ins); b = rates(out)
    j = a.index.intersection(b.index)
    if len(j) < 4:
        return 0.0
    x = a.loc[j].to_numpy(); z = b.loc[j].to_numpy()
    if x.std() == 0 or z.std() == 0:
        return 0.0
    return float(np.corrcoef(x, z)[0, 1])
