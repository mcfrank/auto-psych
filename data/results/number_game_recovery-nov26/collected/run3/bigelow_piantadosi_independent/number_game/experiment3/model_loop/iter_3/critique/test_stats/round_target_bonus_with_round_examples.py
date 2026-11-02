# name: round_target_bonus_with_round_examples
# description: Among sets containing at least one multiple of 10, yes rate for round targets (multiples of 10, not examples) minus other non-example targets; observed above null means a round example licenses round targets more strongly than the concept model predicts (the converse of the landmark veto), below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(int).values
    has_round = (np.nan_to_num(ex, nan=1) % 10 == 0).any(1)
    isex = (ex == t[:, None]).any(1)
    r = df["rating"].astype(float).values
    m = has_round & ~isex
    rd = t % 10 == 0
    a = r[m & rd]; b = r[m & ~rd]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
