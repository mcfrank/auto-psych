# name: within_set_target_dispersion
# description: Mean, over sets with at least 5 distinct non-example targets, of the SD across those targets of each target's yes-rate; observed above the null means people's generalization within a set is more all-or-none across targets (sharper concept boundaries) than the model, below means people's gradients are flatter/more graded than the model predicts.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    is_ex = np.any(E == t[:, None], axis=1)
    d = df.loc[~is_ex, ["number_set", "target", "rating"]]
    cm = d.groupby(["number_set", "target"])["rating"].mean().reset_index()
    g = cm.groupby("number_set")["rating"].agg(["std", "size"])
    g = g[g["size"] >= 5]
    if len(g) == 0:
        return 0.0
    return float(g["std"].mean())
