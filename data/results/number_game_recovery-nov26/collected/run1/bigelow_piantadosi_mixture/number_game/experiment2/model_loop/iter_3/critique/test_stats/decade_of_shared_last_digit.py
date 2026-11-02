# name: decade_of_shared_last_digit
# description: Sets of >=2 examples that all end in the same digit d: yes-rate for non-example targets whose tens digit is d but which do not end in d; observed below null_mean means the model over-generalises via "contains digit d" (people read the shared digit as a last-digit rule only), above means it under-generalises.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    isex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    dmin = np.nanmin(np.abs(E - t[:, None]), axis=1)
    ld = np.mod(E, 10)
    same = (n >= 2) & (np.nanmax(ld, axis=1) == np.nanmin(ld, axis=1))
    d = np.nanmin(ld, axis=1)
    m = same & ~isex & (np.floor(t / 10) == d) & (np.mod(t, 10) != d)
    return float(r[m].mean()) if m.sum() > 0 else 0.0
