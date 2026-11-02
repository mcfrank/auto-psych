# name: lag1_response_autocorr
# description: Pearson correlation between a response and the same participant's previous response (consecutive trial_index); observed > model means the model under-produces sequential dependence / response perseveration.
def test_statistic(df):
    t = df["trial_index"].to_numpy()
    r = df["rating"].astype(float).to_numpy()
    ok = t[1:] == t[:-1] + 1
    a, b = r[1:][ok], r[:-1][ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
