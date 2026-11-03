# name: single_example_generalization_yes
# description: Yes rate for non-example targets on one-example sets; observed < model means people generalize less from a single example than the model's broad hypothesis averaging predicts (observed > model: more).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    mn = np.nanmin(ex, axis=1)
    mx = np.nanmax(ex, axis=1)
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    return rate((n == 1) & (dist > 0))
