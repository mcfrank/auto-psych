# name: within_block_late_minus_early
# description: Within participant x set blocks of >=4 trials, yes rate on the 3rd-and-later presentations of that set minus yes rate on its first two presentations; observed != null means people's generalisation for a set drifts as they keep answering about it (observed < null: they narrow with repetition; > null: they broaden), which the model's trial-independent responses miss.
def test_statistic(df):
    pid = (df["trial_index"].values == 0).cumsum()
    g = pd.DataFrame({"pid": pid, "s": df["number_set"].values, "t": df["trial_index"].values, "r": df["rating"].values.astype(float)})
    g = g.sort_values(["pid", "t"])
    g["pos"] = g.groupby(["pid", "s"]).cumcount()
    g["n"] = g.groupby(["pid", "s"])["r"].transform("size")
    g = g[g["n"] >= 4]
    late = g.loc[g["pos"] >= 2, "r"]
    early = g.loc[g["pos"] < 2, "r"]
    if len(late) == 0 or len(early) == 0:
        return 0.0
    return float(late.mean() - early.mean())
