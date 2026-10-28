# name: single_example_distance_corr
# description: Pearson correlation between yes (0/1) and |target - example| on single-example trials (target != example); observed more negative than null means people's single-example generalization falls off with distance more steeply than the model predicts (model too flat/broad), less negative means the model's similarity gradient is too steep.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    one = ~ns.str.contains(",")
    ex = pd.to_numeric(ns.where(one), errors="coerce").to_numpy(float)
    tg = df["target"].to_numpy(float)
    ok = one.to_numpy() & np.isfinite(ex) & (ex != tg)
    if ok.sum() < 3:
        return 0.0
    d = np.abs(tg[ok] - ex[ok])
    y = df["rating"].to_numpy(float)[ok]
    if np.std(d) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(d, y)[0, 1])
