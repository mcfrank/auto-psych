# name: digit_reversal_target_yes
# description: Yes rate for targets that are the digit reversal of a two-digit example (e.g. 38 for 83, non-palindromic, target not itself an example); observed > null means people use a digit-reversal / shared-digits concept the model lacks, observed < null less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    nonex = dist > 0
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    exn = np.nan_to_num(ex, nan=0.0)
    rev = (exn % 10) * 10 + np.floor(exn / 10)
    ok = (exn >= 10) & (exn <= 99) & (exn % 10 != 0) & (rev != exn)
    hit = np.any(ok & (rev == t[:, None]), axis=1)
    return rate(hit & nonex)
