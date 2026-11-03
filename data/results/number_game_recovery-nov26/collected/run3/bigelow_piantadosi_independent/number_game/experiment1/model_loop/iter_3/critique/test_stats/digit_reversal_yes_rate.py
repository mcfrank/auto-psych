# name: digit_reversal_yes_rate
# description: Yes rate for non-example targets whose two digits are the reverse of some example's (e.g. 38 for example 83; single digits treated as 0x); observed > null means people treat digit-reversed numbers as similar more than the model's digit-containment rules predict, observed < null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    rev = (ex % 10) * 10 + (ex // 10)
    inset = ex.eq(t, axis=0).any(axis=1)
    m = rev.eq(t, axis=0).any(axis=1) & ~inset
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float)[m].mean())
