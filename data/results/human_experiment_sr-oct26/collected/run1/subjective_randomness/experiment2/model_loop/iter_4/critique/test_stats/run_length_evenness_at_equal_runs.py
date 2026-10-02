# name: run_length_evenness_at_equal_runs
# description: Among trials where both sequences have the same number of runs (same alternation rate) but different run-length standard deviations, the proportion choosing the sequence whose run lengths are MORE EVEN (lower SD, e.g. HHHTTHHH over HTHHHHHH); observed above the null means people prefer evenly spread runs more than the model's longest-run, final-run and variety terms predict, below means they prefer lopsided run structure more.
def test_statistic(df):
    def sd(s):
        runs = []; cur = 1
        for x, y in zip(s, s[1:]):
            if x == y:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        return len(runs), float(np.std(runs))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: sd(s) for s in u}
    fa = np.array([m[s] for s in df["sequence_a"]]); fb = np.array([m[s] for s in df["sequence_b"]])
    sel = (fa[:, 0] == fb[:, 0]) & (np.abs(fa[:, 1] - fb[:, 1]) > 1e-9)
    if not sel.any():
        return 0.5
    cl = df["chose_left"].to_numpy()
    chose_even = np.where(fa[:, 1] < fb[:, 1], cl, 1 - cl)
    return float(chose_even[sel].mean())
