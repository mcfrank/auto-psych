# name: participant_parity_effect_sd
# description: Standard deviation across participants (k-th row within each trial_index) of their parity-match effect (yes-rate for non-member targets with the same parity as the smallest example minus opposite parity); observed above null means people differ in how much they use parity/rule-based generalisation (individual strategy differences the single-parameter model lacks), below null means they are more homogeneous than sampling allows.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1)
    member = np.nansum(ex == t[:, None], axis=1) > 0
    same = (t % 2) == (lo % 2)
    pid = df.groupby("trial_index").cumcount().to_numpy()
    d = pd.DataFrame({"pid": pid, "same": same, "r": r})[~member]
    m = d.groupby(["pid", "same"])["r"].mean().unstack()
    if m.shape[0] < 2 or m.shape[1] < 2:
        return 0.0
    eff = (m[True] - m[False]).dropna()
    if len(eff) < 2:
        return 0.0
    return float(eff.std(ddof=1))
