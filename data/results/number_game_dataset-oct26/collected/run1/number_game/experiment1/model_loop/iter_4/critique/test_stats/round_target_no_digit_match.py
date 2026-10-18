# name: round_target_no_digit_match
# description: Among targets sharing no final digit with any example, yes rate for targets that are multiples of 5 minus other targets; observed above null means people favour round numbers (a salience/roundness prior) that the final-digit model gives no credit to, below means they disfavour them.
def test_statistic(df):
    s = df["number_set"].astype(str)
    parts = s.str.split(",", expand=True)
    X = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    valid = ~np.isnan(X)
    Xi = np.nan_to_num(X, nan=-1).astype(int)
    match = (valid & ((Xi % 10) == (t % 10)[:, None])).any(axis=1)
    r = df["rating"].astype(float).to_numpy()
    round5 = (t % 5) == 0
    a = r[~match & round5]
    b = r[~match & ~round5]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
