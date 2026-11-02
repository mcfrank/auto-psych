# name: single_example_yes_rate
# description: Yes rate on trials with a single example whose target differs from it; observed > null means people generalise more broadly from one example than the model's size-principle learner predicts (observed < null: more conservatively).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    ok = (n == 1) & (E[:, 0] != t)
    r = df["rating"].to_numpy().astype(float)
    if ok.sum() == 0:
        return 0.0
    return float(r[ok].mean())
