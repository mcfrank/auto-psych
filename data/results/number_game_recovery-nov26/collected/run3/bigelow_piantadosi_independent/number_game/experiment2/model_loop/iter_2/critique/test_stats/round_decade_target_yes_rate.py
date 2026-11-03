# name: round_decade_target_yes_rate
# description: Yes rate for non-example targets that are multiples of 10, on sets containing no multiple of 10, minus the yes rate for other non-example targets on those sets; observed above null means round numbers are spuriously salient as yes answers beyond the model's rules, below means they are avoided.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = (~np.isnan(E)).sum(1)
    isex = (E == t[:, None]).any(1)
    return E, t, r, n, isex

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r, n, isex = _ex(df)
    no10 = ~((E % 10 == 0).any(1))
    m = no10 & ~isex
    rd = t % 10 == 0
    return _rate(r, m & rd) - _rate(r, m & ~rd)
