# name: short_vs_long_pair_extremity
# description: Mean |pair choice rate - 0.5| over pairs of length <=5 minus the same over pairs of length >=7; observed above null_mean means people are more decisive on short sequences (relative to long) than the model's length-independent sensitivity predicts, below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    lo = np.where(a < b, a, b); hi = np.where(a < b, b, a)
    chose_lo = np.where(a < b, df["chose_left"], 1 - df["chose_left"])
    t = pd.DataFrame({"lo": lo, "hi": hi, "c": chose_lo, "n": a.str.len().to_numpy()})
    g = t.groupby(["lo", "hi"]).agg(r=("c", "mean"), n=("n", "first"))
    ext = (g["r"] - 0.5).abs()
    s = ext[g["n"] <= 5]; l = ext[g["n"] >= 7]
    if len(s) == 0 or len(l) == 0:
        return 0.0
    return float(s.mean() - l.mean())
