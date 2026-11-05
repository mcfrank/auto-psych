# name: within_set_item_sd
# description: Average, over sets tested with >=9 distinct targets, of the SD across that set's targets of per-target yes-rate; observed > null means people discriminate among targets of the same set more sharply than the model, observed < null means people's responses within a set are flatter (more uniform) than the model's graded similarity predicts.
def test_statistic(df):
    it = df.groupby(["number_set", "target"])["rating"].mean().reset_index()
    g = it.groupby("number_set")["rating"]
    sd = g.std(ddof=0)
    n = g.size()
    v = sd[n >= 9].values.astype(float)
    return float(np.mean(v)) if len(v) else 0.0
