# name: set_size_yes_drop
# description: Yes rate on single-example sets minus yes rate on sets of 3-4 examples; observed > model means people narrow their generalization with more examples (size-principle-like) more than the normalized exemplar familiarity predicts.
def test_statistic(df):
    k = (df["number_set"].astype(str).str.count(",") + 1).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    a, b = k == 1, k >= 3
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
