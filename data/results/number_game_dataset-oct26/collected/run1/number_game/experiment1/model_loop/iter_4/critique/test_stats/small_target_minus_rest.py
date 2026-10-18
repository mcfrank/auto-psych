# name: small_target_minus_rest
# description: Yes rate for targets <= 20 minus yes rate for targets > 20 (all sets); observed below null means people reject small targets more than the magnitude-blind digit model predicts (a target-magnitude bias the model lacks), above means they favour them.
def test_statistic(df):
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    a = r[t <= 20]
    b = r[t > 20]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
