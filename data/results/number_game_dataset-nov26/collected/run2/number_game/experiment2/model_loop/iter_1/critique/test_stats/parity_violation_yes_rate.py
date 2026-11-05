# name: parity_violation_yes_rate
# description: Yes-rate on targets of the opposite parity to a multi-example set whose examples all share one parity but not one final digit; observed above null means people honour the shared parity less than the model predicts (model over-weights parity), below means parity is a harder constraint than modelled.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    P = np.mod(E, 2)
    allodd = np.nansum(P == 1, axis=1) == n
    alleven = np.nansum(P == 0, axis=1) == n
    D = np.mod(E, 10)
    dmin = np.nanmin(D, axis=1); dmax = np.nanmax(D, axis=1)
    samedig = dmin == dmax
    m = (n >= 2) & (~samedig) & ((allodd & (t % 2 == 0)) | (alleven & (t % 2 == 1)))
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
