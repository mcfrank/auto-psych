# name: common_last_digit_yes_rate
# description: Yes-rate for targets (not in the set) whose last digit equals the last digit shared by all 2+ examples of the set; observed < null means people generalize a shared last digit less confidently than the model's near-certain "ends in d" rule, observed > null means more.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    return ex, t

def test_statistic(df):
    ex, t = _parse(df)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dig = np.where(valid, np.nan_to_num(ex) % 10, -1)
    first = dig[:, 0]
    all_same = ((dig == first[:, None]) | ~valid).all(axis=1)
    in_set = (valid & (np.nan_to_num(ex) == t[:, None])).any(axis=1)
    m = (n >= 2) & all_same & (t % 10 == first) & ~in_set
    return float(r[m].mean()) if m.any() else 0.0
