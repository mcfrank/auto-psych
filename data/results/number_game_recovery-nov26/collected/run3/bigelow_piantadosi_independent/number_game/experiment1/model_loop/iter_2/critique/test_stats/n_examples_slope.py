# name: n_examples_slope
# description: Yes rate on 1-example sets minus yes rate on 4-example sets (non-example targets); observed > null means people's generalization narrows with more examples more than the model's (or the model is too broad for single examples), observed < null means the model narrows too much.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    n_ex = (s.str.count(",") + 1).values
    vals = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    is_ex = np.any(vals == t[:, None], axis=1)
    a = (n_ex == 1) & ~is_ex
    b = (n_ex == 4) & ~is_ex
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
