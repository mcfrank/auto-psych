# name: doublet_run_choice
# description: Among trials whose sequences differ in their number of runs of length exactly 2 (HH or TT bounded by the other face), the proportion choosing the sequence with MORE doublets; observed above null_mean means people treat short pairs as a hallmark of randomness more than the model predicts, below means they penalise them.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    def dbl(s):
        u = pd.unique(s)
        def f(x):
            runs, cur = [], 1
            for i in range(1, len(x)):
                if x[i] == x[i - 1]:
                    cur += 1
                else:
                    runs.append(cur); cur = 1
            runs.append(cur)
            return sum(1 for r in runs if r == 2)
        return s.map({x: f(x) for x in u}).to_numpy()
    da, db = dbl(a), dbl(b)
    mask = da != db
    if not mask.any():
        return 0.5
    c = df["chose_left"].to_numpy()
    chose_more = np.where(da > db, c, 1 - c)
    return float(chose_more[mask].mean())
