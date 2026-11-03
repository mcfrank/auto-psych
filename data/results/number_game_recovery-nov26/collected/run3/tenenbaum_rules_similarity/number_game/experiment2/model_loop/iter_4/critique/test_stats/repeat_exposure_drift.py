# name: repeat_exposure_drift
# description: For participant-set pairs answered >=10 times (participants delimited by trial_index resetting to 0), yes rate on the 6th and later encounters of that set minus the first 5; observed > null means people become more liberal with a set the more they are queried about it (the model has no within-session dynamics), observed < null means they become stricter.
def test_statistic(df):
    pid = (df["trial_index"].astype(int).values == 0).cumsum()
    sets = df["number_set"].astype(str).values
    r = df["rating"].astype(float).values
    key = pd.Series(pid.astype(str)) + "|" + pd.Series(sets)
    k = key.groupby(key).cumcount().values
    tot = key.map(key.value_counts()).values
    m = tot >= 10
    a = r[m & (k >= 5)]
    b = r[m & (k < 5)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
