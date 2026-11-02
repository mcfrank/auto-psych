# name: proximity_gradient_nonmembers
# description: Among non-member targets, yes rate for targets within 3 of the nearest example minus yes rate for targets more than 20 away; observed below null means the model predicts a steeper similarity/distance gradient than people show.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    dist = s.sub(t, axis=0).abs().min(axis=1).values
    r = df["rating"].values.astype(float)
    near = (dist > 0) & (dist <= 3)
    far = dist > 20
    a = r[near].mean() if near.any() else 0.0
    b = r[far].mean() if far.any() else 0.0
    return float(a - b)
