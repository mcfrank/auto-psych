# name: near_far_similarity_gradient
# description: Yes rate for targets within 3 of the nearest example minus yes rate for targets 15+ away from every example (excluding targets equal to an example); observed > null means people show more graded nearby-number generalization than the sharpened model produces.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    r = df["rating"].astype(float).to_numpy()
    near = (d >= 1) & (d <= 3)
    far = d >= 15
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
