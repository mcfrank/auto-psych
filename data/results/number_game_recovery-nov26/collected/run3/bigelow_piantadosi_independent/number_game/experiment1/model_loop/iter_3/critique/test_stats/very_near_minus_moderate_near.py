# name: very_near_minus_moderate_near
# description: Yes rate for non-example targets 1-3 from the nearest example minus yes rate for targets 4-10 away; the model's intervals imply a monotone decline (positive value), so observed < null means people generalise relatively less to immediate neighbours than to moderately near numbers, observed > null means a steeper local gradient.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    dist = ex.sub(t, axis=0).abs().min(axis=1)
    r = df["rating"].astype(float)
    a = r[(dist >= 1) & (dist <= 3)]
    b = r[(dist >= 4) & (dist <= 10)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
