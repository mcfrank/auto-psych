# name: digit_reversal_target_yes_gap
# description: Yes rate for non-example two-digit targets that are the digit reversal of some example (e.g. 27 after 72, excluding palindromes) minus other non-example targets on sets with at least one non-palindromic two-digit example; observed above null means people accept digit-reversed numbers beyond what the model's concepts imply, below null means they reject them more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    n = (~np.isnan(ex)).sum(1)
    mx = np.nanmax(ex, 1); mn = np.nanmin(ex, 1)
    isex = (ex == t[:, None]).any(1)
    e = np.nan_to_num(ex, nan=-1).astype(int)
    two = (e >= 10) & (e <= 99) & (e % 10 != e // 10) & (e % 10 != 0)
    rev = np.where(two, (e % 10) * 10 + e // 10, -999)
    hit = (rev == t[:, None]).any(1) & ~isex
    elig = two.any(1) & ~isex
    a = r[hit]; b = r[elig & ~hit]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
