# name: participant_set_block_sd
# description: For each set seen by >=2 participants with >=4 trials each (participant = run of rows starting at trial_index 0), SD across participants of their yes rate on that set, averaged over sets; observed > null means people hold a persistent personal concept guess per set (responses cluster within participant x set) that the model, which redraws independently per trial, under-produces.
def test_statistic(df):
    pid = (df["trial_index"].values == 0).cumsum()
    g = pd.DataFrame({"pid": pid, "s": df["number_set"].values, "r": df["rating"].values.astype(float)})
    b = g.groupby(["s", "pid"])["r"].agg(["mean", "size"]).reset_index()
    b = b[b["size"] >= 4]
    sd = b.groupby("s")["mean"].agg(["std", "size"])
    sd = sd[sd["size"] >= 2]["std"]
    return float(sd.mean()) if len(sd) else 0.0
