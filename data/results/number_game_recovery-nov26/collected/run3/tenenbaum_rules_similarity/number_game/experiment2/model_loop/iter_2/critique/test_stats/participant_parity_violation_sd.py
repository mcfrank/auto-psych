# name: participant_parity_violation_sd
# description: Across participants (session = run of trial_index from 0), SD of each participant's yes rate on in-span non-example targets whose parity differs from all-same-parity examples (sets >=2); observed > null means people differ in whether they adopt a range or a parity rule more than the model's single shared posterior allows, observed < null less.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    pid = np.cumsum(df["trial_index"].to_numpy() == 0)
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    lo, hi = np.nanmin(E, axis=1), np.nanmax(E, axis=1)
    par = np.where(valid, np.mod(np.nan_to_num(E), 2), np.nan)
    pmin, pmax = np.nanmin(par, axis=1), np.nanmax(par, axis=1)
    m = (n >= 2) & (pmin == pmax) & (t > lo) & (t < hi) & (np.mod(t, 2) != pmin)
    if m.sum() == 0:
        return 0.0
    g = pd.Series(r[m]).groupby(pid[m]).mean()
    return float(g.std(ddof=0)) if len(g) > 1 else 0.0
