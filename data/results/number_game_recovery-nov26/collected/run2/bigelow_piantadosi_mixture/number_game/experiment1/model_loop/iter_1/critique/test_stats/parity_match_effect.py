# name: parity_match_effect
# description: For multi-example sets whose examples all share parity, yes rate for targets of the same parity minus targets of the opposite parity (targets in set excluded); observed above null means people use parity more than the model, below means the model over-produces parity-based generalization.
def test_statistic(df):
    s = df["number_set"].astype(str)
    n = (s.str.count(",") + 1).to_numpy()
    ex = s.str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    par = np.mod(np.where(valid, ex, 0), 2)
    n_odd = (par * valid).sum(axis=1)
    all_even = (n_odd == 0)
    all_odd = (n_odd == n)
    t = df["target"].astype(int).to_numpy()
    in_set = (ex == t[:, None]).any(axis=1)
    base = (n >= 2) & (all_even | all_odd) & ~in_set
    set_par = np.where(all_odd, 1, 0)
    same = base & ((t % 2) == set_par)
    diff = base & ((t % 2) != set_par)
    y = df["rating"].to_numpy().astype(float)
    if same.sum() == 0 or diff.sum() == 0:
        return 0.0
    return float(y[same].mean() - y[diff].mean())
