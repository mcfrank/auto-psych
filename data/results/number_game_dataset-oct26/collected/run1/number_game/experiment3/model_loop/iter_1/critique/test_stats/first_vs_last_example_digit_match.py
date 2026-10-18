# name: first_vs_last_example_digit_match
# description: Among multi-example sets whose first and last shown examples have different final digits, yes-rate for targets sharing the first example's final digit (but not the last's) minus yes-rate for targets sharing the last's (but not the first's); nonzero observed vs a ~0 null means people weight examples by presentation order, which the order-blind model cannot do (positive = primacy, negative = recency).
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",")
    n = parts.str.len()
    f = parts.str[0].astype(int) % 10
    l = parts.str[-1].astype(int) % 10
    td = df["target"].astype(int) % 10
    base = (n >= 2) & (f != l)
    r = df["rating"].astype(float)
    a = r[base & (td == f)]
    b = r[base & (td == l)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
