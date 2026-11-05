# name: single_example_generalization_rate
# description: Yes rate for non-example targets on sets with exactly one example; observed above null means people generalize more broadly from a single example than the model's size principle allows (below = more conservative).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    t = df["target"].astype(int).to_numpy()
    n_ex = np.sum(~np.isnan(ex), axis=1)
    is_ex = (ex == t[:, None]).any(axis=1)
    m = (n_ex == 1) & ~is_ex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
