# name: trial_index_yes_slope
# description: Pearson correlation between trial_index and the yes response; the model has no time dependence (null near 0), so an observed positive value means people drift toward saying yes more later in the session, which the model fails to produce.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
