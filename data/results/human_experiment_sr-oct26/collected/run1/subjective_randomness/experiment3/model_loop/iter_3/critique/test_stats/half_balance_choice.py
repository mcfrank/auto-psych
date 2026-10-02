# name: half_balance_choice
# description: Among pairs whose two sequences differ in half-imbalance (|#H in first half - #H in second half|), rate of choosing the sequence whose heads are more evenly split between its halves; observed above null_mean means people penalise heads/tails clumping into one half more than the model's global-imbalance and lopsided-stretch terms predict, below means less.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    def hb(s):
        L = s.str.len().to_numpy()
        h = L // 2
        x = s.to_numpy().astype(str)
        arr = (np.array([list(v.ljust(8, " ")) for v in x]) == "H").astype(int)
        cs = np.concatenate([np.zeros((len(x), 1), int), arr.cumsum(1)], 1)
        first = cs[np.arange(len(x)), h]
        total = cs[np.arange(len(x)), L]
        second = total - cs[np.arange(len(x)), L - h]
        return np.abs(first - second)
    da = hb(a); db = hb(b)
    mask = da != db
    cl = df["chose_left"].to_numpy().astype(float)
    chose_even = np.where(da < db, cl, 1 - cl)[mask]
    return float(chose_even.mean()) if len(chose_even) else 0.5
