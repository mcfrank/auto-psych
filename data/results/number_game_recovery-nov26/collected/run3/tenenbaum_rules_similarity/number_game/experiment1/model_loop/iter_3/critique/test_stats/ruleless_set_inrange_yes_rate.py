# name: ruleless_set_inrange_yes_rate
# description: Among multi-example sets with no shared arithmetic structure (examples not all the same parity, not sharing a last digit, gcd < 3), the yes-rate for non-example targets strictly inside the examples' [min,max]; observed above model means people fill in the magnitude range of structureless examples more than the model's Erlang interval prior predicts, below means less.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    ex = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    present = ~np.isnan(ex)
    n_ex = present.sum(axis=1)
    exi = np.nan_to_num(ex, nan=0.0).astype(int)
    g = np.gcd.reduce(np.where(present, exi, 0), axis=1)
    par = exi % 2
    same_par = np.all(~present | (par == par[:, [0]]), axis=1)
    ld = exi % 10
    same_ld = np.all(~present | (ld == ld[:, [0]]), axis=1)
    t = df["target"].astype(int).to_numpy()
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    mask = (n_ex >= 2) & ~same_par & ~same_ld & (g < 3) & (t > lo) & (t < hi) & ~is_ex
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[mask].mean())
