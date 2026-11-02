# name: digit_reversal_target_yes
# description: Yes rate on two-digit targets (not ending in 0, not themselves examples) whose digit reversal is one of the shown examples (e.g. 38 after 83); observed > null means people treat digit-swapped numbers as related more than the model's rule/range concepts imply, observed < null means less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    ex = sets.str.split(",")
    tt = pd.Series(t).astype(str)
    rev = tt.str[::-1].str.lstrip("0")
    ok = (t >= 10) & (t % 10 != 0) & (t < 100)
    hit = np.array([b in e and a not in e for a, b, e in zip(tt, rev, ex)])
    m = ok & hit
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
