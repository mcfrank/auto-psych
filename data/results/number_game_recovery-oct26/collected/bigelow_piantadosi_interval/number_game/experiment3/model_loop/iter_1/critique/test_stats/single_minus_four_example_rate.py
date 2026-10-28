# name: single_minus_four_example_rate
# description: Among non-example targets, yes rate after 1-example sets minus yes rate after 4-example sets; observed above null_mean means people narrow their generalization with more examples more strongly than the model (model's set-size effect too weak), below means the model over-narrows.
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None]
    is_ex = (ex == t).any(axis=1)
    n = (s.str.count(",") + 1).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    a = (n == 1) & ~is_ex
    b = (n == 4) & ~is_ex
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
