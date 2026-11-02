# name: participant_far_generalization_sd
# description: SD across participants (new participant at each trial_index reset) of the yes-rate on non-example targets 10+ outside the examples' span, after subtracting the mean of the participant's design batch (5 consecutive participants); observed above null_mean means individuals differ in generalization breadth beyond what the single-population model allows.
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy()
    pid = np.cumsum(np.r_[True, ti[1:] <= ti[:-1]]) - 1
    t = df["target"].astype(int).to_numpy()
    u = df["number_set"].unique()
    lo = df["number_set"].map({s: min(int(x) for x in s.split(",")) for s in u}).to_numpy()
    hi = df["number_set"].map({s: max(int(x) for x in s.split(",")) for s in u}).to_numpy()
    far = (t <= lo - 10) | (t >= hi + 10)
    r = df["rating"].astype(float).to_numpy()
    if far.sum() == 0:
        return 0.0
    rates = pd.Series(r[far]).groupby(pid[far]).mean()
    batch = rates.index.to_numpy() // 5
    dev = rates.to_numpy() - pd.Series(rates.to_numpy()).groupby(batch).transform("mean").to_numpy()
    return float(np.std(dev)) if len(dev) > 1 else 0.0
