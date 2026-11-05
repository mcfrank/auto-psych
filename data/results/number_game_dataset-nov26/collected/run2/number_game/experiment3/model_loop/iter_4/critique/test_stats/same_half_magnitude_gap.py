# name: same_half_magnitude_gap
# description: On 2-4-example sets whose examples all lie in one half of 1-100 (all <=50 or all >50), yes rate for targets in that same half minus targets in the other half; observed > null means people treat 'small numbers'/'big numbers' as a category more sharply than the model's distance band (observed < null: the model over-sharpens magnitude).
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    K = len(u)
    half = np.full(K, -1)
    for i, s in enumerate(u):
        e = [int(x) for x in s.split(",")]
        if len(e) >= 2:
            if all(x <= 50 for x in e):
                half[i] = 0
            elif all(x > 50 for x in e):
                half[i] = 1
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    h = half[inv]
    th = (t > 50).astype(int)
    a = (h >= 0) & (th == h)
    b = (h >= 0) & (th != h)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
