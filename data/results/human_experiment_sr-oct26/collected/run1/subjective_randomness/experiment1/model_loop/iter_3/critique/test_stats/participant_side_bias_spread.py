# name: participant_side_bias_spread
# description: SD across participants of each participant's rate of choosing Left; observed above null means some people have a response-side bias that the model (symmetric guessing) does not produce.
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
    g = df.groupby("participant_id")["chose_left"].mean()
    return float(g.std(ddof=0)) if len(g) > 1 else 0.0
