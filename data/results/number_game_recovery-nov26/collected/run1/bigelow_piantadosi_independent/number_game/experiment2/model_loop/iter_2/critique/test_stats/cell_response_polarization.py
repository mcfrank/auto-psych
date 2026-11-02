# name: cell_response_polarization
# description: Mean over (set, target) cells with >= 3 responses of |cell yes rate - 0.5|; observed > null means people agree more sharply (cells closer to all-yes / all-no) than the model's moderate predicted probabilities produce, observed < null means people are less consensual than the model.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    nonex = dist > 0
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    g = pd.DataFrame({"k": key.values, "r": r}).groupby("k")["r"].agg(["mean", "size"])
    g = g[g["size"] >= 3]
    return float(np.abs(g["mean"] - 0.5).mean()) if len(g) else 0.0
