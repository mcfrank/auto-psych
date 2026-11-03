# name: shared_last_digit_far_yes_rate
# description: Among sets whose examples all share a last digit, yes rate for targets with that same last digit lying >15 from every example; observed below null_mean means the model still over-extends "ends in d" guesses to distant numbers, above means it under-weights digit rules.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n_ex = np.sum(~np.isnan(E), axis=1)
    ld = np.mod(E, 10)
    first = ld[:, 0]
    same = np.all((ld == first[:, None]) | np.isnan(ld), axis=1) & (n_ex >= 2)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    m = same & (np.mod(t, 10) == first) & (dist > 15)
    return float(r[m].mean()) if m.any() else 0.0
