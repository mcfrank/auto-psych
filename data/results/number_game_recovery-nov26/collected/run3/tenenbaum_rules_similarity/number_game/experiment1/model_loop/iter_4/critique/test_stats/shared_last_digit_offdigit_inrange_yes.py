# name: shared_last_digit_offdigit_inrange_yes
# description: Among sets (>=2 examples) whose examples all end in the same digit, the "yes" rate for targets inside [min,max] of the examples that do NOT end in that digit; observed below null = model over-generalises via (parity-)range concepts when a last-digit rule is evident.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    last = ex % 10
    same = (last.nunique(axis=1) == 1) & (n >= 2)
    d = last.iloc[:, 0]
    t = df["target"].astype(int)
    lo, hi = ex.min(axis=1), ex.max(axis=1)
    m = same & (t % 10 != d) & (t >= lo) & (t <= hi)
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].mean())
