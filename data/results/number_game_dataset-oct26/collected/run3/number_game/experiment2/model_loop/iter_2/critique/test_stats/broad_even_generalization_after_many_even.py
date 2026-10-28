# name: broad_even_generalization_after_many_even
# description: Among sets of 3-4 examples that are all even, the yes-rate on even non-example targets; observed > model means people keep generalizing to the broad "even numbers" concept after several examples more than the model's size-principle posterior (which collapses onto narrow rules like powers of 2) predicts, observed < model means they narrow even more sharply.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    all_even = np.all(np.isnan(ex) | (np.mod(ex, 2) == 0), axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    m = (n >= 3) & all_even & (t % 2 == 0) & ~is_ex
    return float(y[m].mean()) if m.any() else 0.0
