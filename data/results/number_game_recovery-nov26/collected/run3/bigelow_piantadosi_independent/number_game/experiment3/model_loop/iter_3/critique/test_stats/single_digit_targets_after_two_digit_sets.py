# name: single_digit_targets_after_two_digit_sets
# description: Among sets whose examples are all two-digit (10-99), yes rate for single-digit targets (1-9) minus other non-example targets of 10-99; observed below null means people treat single-digit numbers as a category excluded by two-digit examples more than the model predicts, above null means they accept them more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    e = np.nan_to_num(ex, nan=50)
    two = ((e >= 10) & (e <= 99)).all(1)
    t = df["target"].astype(int).values
    isex = (ex == t[:, None]).any(1)
    r = df["rating"].astype(float).values
    m = two & ~isex
    a = r[m & (t <= 9)]; b = r[m & (t >= 10) & (t <= 99)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
