# name: digit_reversal_target_contrast
# description: Yes rate for non-member targets whose digits are the reverse of some example (e.g. 72 after 27, two-digit, non-palindromic), minus the yes rate for all other non-member targets in the same sets; observed above null means people generalise to digit reversals more than the model (which has no digit-order concepts) predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    exr = np.where(np.isnan(ex) | (ex < 10) | (ex > 99) | (ex % 11 == 0), np.nan, (ex % 10) * 10 + ex // 10)
    rev = np.any(exr == t[:, None], axis=1) & ~member
    has = np.any(~np.isnan(exr) & (exr >= 10), axis=1)
    a = rev; b = has & ~rev & ~member
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
