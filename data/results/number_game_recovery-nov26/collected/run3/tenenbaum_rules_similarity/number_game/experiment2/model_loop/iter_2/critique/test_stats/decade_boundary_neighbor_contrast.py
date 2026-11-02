# name: decade_boundary_neighbor_contrast
# description: Among non-example targets 1-5 away from their nearest example, yes rate when the target shares that example's tens digit minus when it lies in a different decade; observed > null means people treat decades as category boundaries (a structure the model's decade-blind intervals cannot produce), observed < null the reverse.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    D = np.abs(E - t[:, None])
    D = np.where(np.isnan(D), np.inf, D)
    j = np.argmin(D, axis=1)
    near = D[np.arange(len(t)), j]
    ex = E[np.arange(len(t)), j]
    base = (near >= 1) & (near <= 5) & (t < 100) & (ex < 100)
    same = base & (np.floor(t / 10) == np.floor(ex / 10))
    diff = base & (np.floor(t / 10) != np.floor(ex / 10))
    if same.sum() == 0 or diff.sum() == 0:
        return 0.0
    return float(r[same].mean() - r[diff].mean())
