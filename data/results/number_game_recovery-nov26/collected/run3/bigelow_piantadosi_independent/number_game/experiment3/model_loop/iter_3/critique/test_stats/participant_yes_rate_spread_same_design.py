# name: participant_yes_rate_spread_same_design
# description: SD of participant yes rates around the mean of participants who saw the same stimuli (participants delimited by trial_index restarts, design group = first set seen); observed above null means real individual differences in yes-bias the pooled model lacks, below null means participants are more homogeneous than binomial noise.
def test_statistic(df):
    ti = df["trial_index"].astype(int).values
    pid = np.cumsum(np.r_[True, np.diff(ti) != 1])
    d = pd.DataFrame({"pid": pid, "r": df["rating"].astype(float).values, "s": df["number_set"].astype(str).values})
    p = d.groupby("pid").agg(r=("r", "mean"), g=("s", "first"))
    dev = p["r"] - p.groupby("g")["r"].transform("mean")
    return float(np.sqrt(np.mean(dev.values ** 2)))
