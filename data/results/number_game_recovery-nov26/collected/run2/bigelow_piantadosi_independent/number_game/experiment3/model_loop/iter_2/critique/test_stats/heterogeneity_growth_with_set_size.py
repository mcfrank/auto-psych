# name: heterogeneity_growth_with_set_size
# description: Between-participant SD (participants = trial_index resets; each participant's rate minus their 5-person design batch mean) of the yes-rate on out-of-span non-example targets in 3-4-example sets, minus the same SD for 1-example sets; observed above null_mean means individual differences grow with set size more than the model's per-person range looseness produces (below = the model over-predicts that growth).
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.concatenate([[0], np.cumsum(np.diff(ti) < 0)])
    batch = pid // 5
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    out = (t < np.nanmin(ex, axis=1)) | (t > np.nanmax(ex, axis=1))
    y = df["rating"].astype(float).to_numpy()
    def sd(mask):
        d = pd.DataFrame({"p": pid[mask], "b": batch[mask], "y": y[mask]})
        pr = d.groupby(["b", "p"])["y"].mean().reset_index()
        if len(pr) < 3:
            return 0.0
        pr["dev"] = pr["y"] - pr.groupby("b")["y"].transform("mean")
        return float(np.sqrt(np.mean(pr["dev"] ** 2)))
    return sd(out & (n >= 3)) - sd(out & (n == 1))
