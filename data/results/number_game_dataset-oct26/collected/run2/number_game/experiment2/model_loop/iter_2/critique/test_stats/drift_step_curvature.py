# name: drift_step_curvature
# description: (yes-rate in trials 150-299 minus trials 0-149) minus (yes-rate in trials 300+ minus trials 150-299); positive observed above the null means the session drift toward "yes" is a front-loaded step/saturating jump rather than the steady linear relaxation the model assumes (below means the drift accelerates late).
def test_statistic(df):
    ti = df["trial_index"].astype(float).values
    y = df["rating"].astype(float).values
    e, m, l = ti < 150, (ti >= 150) & (ti < 300), ti >= 300
    if e.sum() == 0 or m.sum() == 0 or l.sum() == 0:
        return 0.0
    return float((y[m].mean() - y[e].mean()) - (y[l].mean() - y[m].mean()))
