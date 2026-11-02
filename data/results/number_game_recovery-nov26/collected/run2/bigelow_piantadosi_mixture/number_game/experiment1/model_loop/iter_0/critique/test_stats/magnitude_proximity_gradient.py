# name: magnitude_proximity_gradient
# description: For non-example targets, yes rate when the target is within 5 of the nearest example minus yes rate when it is more than 20 from every example; observed above null means people generalise by numeric closeness more strongly than the model's interval hypotheses allow.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    tgt = df["target"].astype(int).to_numpy()
    d = np.nanmin(np.abs(ex - tgt[:, None]), axis=1)
    r = df["rating"].astype(float).to_numpy()
    near = (d > 0) & (d <= 5)
    far = d > 20
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
