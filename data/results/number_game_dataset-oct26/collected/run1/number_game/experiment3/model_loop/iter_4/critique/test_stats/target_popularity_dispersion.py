# name: target_popularity_dispersion
# description: Standard deviation across target numbers (pooled over all sets, targets with >=27 responses) of each target's overall yes-rate; observed > null means some numbers are intrinsically favoured/disfavoured as targets (a set-independent target prior) beyond what the model's set-target features predict; observed < null means the model over-differentiates targets.
def test_statistic(df):
    g = df.groupby(df["target"].astype(int))["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 27]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
