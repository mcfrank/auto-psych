# name: trial_drift_consensus
# description: Agreement with each pair's majority choice in the second half of each participant's trials minus the first half; observed nonzero beyond null means judgments sharpen (positive) or degrade (negative) over the session, which the model's time-invariant lapse and ideal cannot produce.
def _alt(s):
    u = pd.unique(s)
    m = {x: sum(a != b for a, b in zip(x, x[1:])) / (len(x) - 1) for x in u}
    return s.map(m).to_numpy(dtype=float)


def _imb(s):
    return np.abs(s.str.count("H").to_numpy() / s.str.len().to_numpy() - 0.5)


def _maxrun(s):
    u = pd.unique(s)
    def mr(x):
        best = cur = 1
        for a, b in zip(x, x[1:]):
            cur = cur + 1 if a == b else 1
            best = max(best, cur)
        return best
    return s.map({x: mr(x) for x in u}).to_numpy(dtype=float)



def test_statistic(df):
    a, b = df["sequence_a"].to_numpy(), df["sequence_b"].to_numpy()
    y = df["chose_left"].to_numpy()
    first = np.where(a < b, a, b)
    second = np.where(a < b, b, a)
    cf = np.where(a < b, y, 1 - y)
    key = (pd.Series(first) + "|" + pd.Series(second)).to_numpy()
    pm_ = pd.Series(cf).groupby(key).transform("mean").to_numpy()
    agree = np.where(pm_ >= 0.5, cf, 1 - cf)
    t = df["trial_index"].to_numpy()
    med = df.groupby("participant_id")["trial_index"].transform("median").to_numpy()
    late = t > med
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(agree[late].mean() - agree[~late].mean())
