# name: short_minus_long_pair_consensus
# description: Mean pair-level consensus strength |share choosing one sequence - 0.5| over distinct pairs of length 2-5 minus the same over pairs of length 8; observed above null_mean means people are more decisive on short sequences (relative to long) than the model's length-normalised features predict, below means less decisive on short ones.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    sw = a < b
    key = np.char.add(np.char.add(np.where(sw, a, b), "|"), np.where(sw, b, a))
    cl = df["chose_left"].to_numpy().astype(float)
    cf = np.where(sw, cl, 1 - cl)
    L = np.char.str_len(a)
    g = pd.DataFrame({"k": key, "c": cf, "L": L}).groupby("k").agg(c=("c", "mean"), L=("L", "first"))
    ext = np.abs(g["c"].to_numpy() - 0.5)
    Lg = g["L"].to_numpy()
    s = ext[Lg <= 5]; l = ext[Lg == 8]
    if len(s) == 0 or len(l) == 0:
        return 0.0
    return float(s.mean() - l.mean())
