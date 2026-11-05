# name: any_example_digit_overlap_effect
# description: In multi-example sets, yes-rate for targets sharing at least one digit with at least one example (surface similarity) minus targets sharing no digit with any example; the model only uses digits common to all examples, so observed > null means people use partial surface digit similarity the model ignores (observed < null: the reverse).
def test_statistic(df):
    s = df["number_set"].astype(str)
    n = (s.str.count(",") + 1).values
    t = df["target"].astype(int).astype(str)
    digs = s.str.replace(",", "", regex=False)
    share = np.zeros(len(df), bool)
    for d in "0123456789":
        share |= (t.str.contains(d, regex=False).values & digs.str.contains(d, regex=False).values)
    r = df["rating"].astype(float).values
    a = (n >= 2) & share
    b = (n >= 2) & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
