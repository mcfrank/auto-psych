# name: within_set_target_spread
# description: Mean over number sets (with 3+ distinct targets) of the standard deviation across targets of the per-(set,target) yes-rate; observed above the null means people's generalization is more all-or-none across targets (sharper, rule-like gradients) than the model predicts, below means people are flatter/more uniform than the model.
def test_statistic(df):
    d = df[["number_set", "target"]].copy()
    d["y"] = df["rating"].astype(float).values
    cell = d.groupby(["number_set", "target"])["y"].mean().reset_index()
    g = cell.groupby("number_set")["y"]
    sd = g.std(ddof=0)
    n = g.size()
    sd = sd[n >= 3]
    if len(sd) == 0:
        return 0.0
    return float(sd.mean())
