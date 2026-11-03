# name: shared_tens_digit_yes_rate
# description: Yes rate for non-example targets sharing the tens digit (same decade, e.g. 42 vs 47) with some example but not its last digit; observed above null means people use decade/leading-digit similarity the model lacks.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return E, t, r

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r = _ex(df)
    isex = (E == t[:, None]).any(1)
    tens = (np.floor(E / 10) == np.floor(t / 10)[:, None]).any(1)
    last = ((E % 10) == (t % 10)[:, None]).any(1)
    m = tens & ~last & ~isex & (t < 100)
    return _rate(r, m)
