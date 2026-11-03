# name: trial_index_yes_slope
# description: Pearson correlation between trial_index and rating; the model has no time dependence (null ~0), so observed > null means people say "yes" more as the session goes on, observed < null means they grow more conservative.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
