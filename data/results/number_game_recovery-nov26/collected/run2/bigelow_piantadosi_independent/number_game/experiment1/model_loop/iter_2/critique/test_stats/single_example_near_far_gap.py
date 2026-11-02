# name: single_example_near_far_gap
# description: In one-example sets, yes rate for targets 1-10 from the example minus yes rate for targets >10 away; t_observed above null_mean means people's generalization from a single example is more local (similarity-based) than the model predicts, below means flatter/more rule-based.
def test_statistic(df):
    s = df["number_set"].astype(str)
    n = s.str.count(",").to_numpy() + 1
    x = pd.to_numeric(s.str.split(",").str[0], errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    d = np.abs(t - x)
    r = df["rating"].to_numpy(dtype=float)
    a = (n == 1) & (d >= 1) & (d <= 10); b = (n == 1) & (d > 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
