# name: drift_step_curvature
# description: (yes-rate in trials 150-299 minus trials 0-149) minus (yes-rate in trials 300+ minus trials 150-299); observed above the null means the session drift toward "yes" is a front-loaded / saturating jump rather than the linear criterion relaxation the model assumes (below means the drift accelerates late).
def test_statistic(df):
    ti = df["trial_index"].astype(float).values
    y = df["rating"].astype(float).values
    a, b, c = ti < 150, (ti >= 150) & (ti < 300), ti >= 300
    if a.sum() == 0 or b.sum() == 0 or c.sum() == 0:
        return 0.0
    ma, mb, mc = y[a].mean(), y[b].mean(), y[c].mean()
    return float((mb - ma) - (mc - mb))
