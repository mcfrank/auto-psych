# name: target_in_set_echo_rate
# description: Yes-rate for trials whose target is itself one of the shown examples (should be ~1 under the model); observed < null means people reject shown examples more than the model allows (lapse/noise missing).
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).astype(str).values
    key = pd.Series(sets) + "|" + pd.Series(t)
    uk = key.unique()
    lut = {k: (k.split("|")[1] in k.split("|")[0].split(",")) for k in uk}
    m = key.map(lut).values.astype(bool)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
