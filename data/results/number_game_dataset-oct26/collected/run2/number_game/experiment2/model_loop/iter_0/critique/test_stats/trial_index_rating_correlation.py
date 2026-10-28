# name: trial_index_rating_correlation
# description: Pearson correlation between trial_index and rating; observed above the null means people drift toward saying yes more over the session, a time trend the stationary model does not produce (below means drift toward no).
def test_statistic(df):
    x = df["trial_index"].astype(float).values
    y = df["rating"].astype(float).values
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
