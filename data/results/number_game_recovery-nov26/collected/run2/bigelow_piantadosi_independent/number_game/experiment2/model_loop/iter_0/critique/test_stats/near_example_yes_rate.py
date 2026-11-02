# name: near_example_yes_rate
# description: Yes rate for targets within 1-3 of the nearest example (but not an example itself); observed above null means the model under-generalizes to close neighbours (too little similarity gradient), below means it over-generalizes.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    return rate((d >= 1) & (d <= 3))
