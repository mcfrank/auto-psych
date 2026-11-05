# name: shared_digit_nonmember_late_minus_early
# description: For sets of >=2 examples all sharing a last digit, yes rate for targets lacking that digit in the second half of the session (trial_index above median) minus the first half; the model is stationary (null ~0), so observed below null means people become stricter about digit rules over the session, above null more lenient.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    last = ex % 10
    uni = (last.max(axis=1) == last.min(axis=1)) & (n >= 2)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    non = uni & ((t % 10) != last.iloc[:, 0])
    ti = df["trial_index"].astype(float)
    late = ti > ti.median()
    a = non & late; b = non & ~late
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
