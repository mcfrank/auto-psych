# name: single_example_parity_contrast
# description: Among one-example sets, yes-rate for targets sharing the example's parity minus yes-rate for opposite-parity targets; observed above null means the model under-produces parity generalisation from a single example.
def test_statistic(df):
    s = df["number_set"].astype(str)
    m = (s.str.count(",") == 0).values
    if m.sum() == 0:
        return 0.0
    ex = pd.to_numeric(s[m], errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values[m]
    r = df["rating"].astype(float).values[m]
    same = (ex % 2) == (t % 2)
    a = r[same].mean() if same.any() else 0.0
    b = r[~same].mean() if (~same).any() else 0.0
    return float(a - b)
