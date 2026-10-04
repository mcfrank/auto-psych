# name: short_vs_long_item_strength
# description: Mean item consensus |pair choice rate - 0.5| over pairs of length <=4 minus that over pairs of length >=7; observed above null means short pairs are judged more decisively relative to long ones than the model's length-power sensitivity allows (model under-produces short-pair consensus), below means long pairs are more decisive than modelled.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    first = a < b
    key = np.where(first, a + "|" + b, b + "|" + a)
    y = np.where(first, df["chose_left"], 1 - df["chose_left"])
    g = pd.DataFrame({"k": key, "y": y, "L": a.str.len().to_numpy()}).groupby("k").agg(y=("y", "mean"), L=("L", "first"))
    s = (g["y"] - 0.5).abs()
    sh = s[g["L"] <= 4]; lo = s[g["L"] >= 7]
    if len(sh) == 0 or len(lo) == 0:
        return 0.0
    return float(sh.mean() - lo.mean())
