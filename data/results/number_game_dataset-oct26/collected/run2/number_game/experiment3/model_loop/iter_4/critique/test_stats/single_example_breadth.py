# name: single_example_breadth
# description: Overall yes-rate on one-example sets minus the overall yes-rate on sets of 2+ examples; observed > null means people generalize from a single example more broadly, relative to multi-example sets, than the model does (model under-produces the size-principle drop from 1 to 2+ examples), observed < null the reverse.
def test_statistic(df):
    k = df["number_set"].astype(str).str.count(",").values + 1
    r = df["rating"].values.astype(float)
    if (k == 1).sum() == 0 or (k >= 2).sum() == 0:
        return 0.0
    return float(r[k == 1].mean() - r[k >= 2].mean())
