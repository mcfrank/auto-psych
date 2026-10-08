# name: balance_preference_similar_switches
# description: Among pairs whose switch counts differ by at most 1 and whose head-count imbalances |#H - n/2| differ, the proportion of choices for the more balanced sequence; observed above null_mean means people weight H/T balance more than the model implies, below means less.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    def switches(s):
        return s.str.count(r"(?=HT)|(?=TH)")
    ka, kb = switches(a), switches(b)
    n = a.str.len()
    ia = (a.str.count("H") - n / 2).abs(); ib = (b.str.count("H") - n / 2).abs()
    m = ((ka - kb).abs() <= 1) & (ia != ib)
    if m.sum() == 0:
        return 0.5
    a_bal = (ia < ib)[m].to_numpy()
    y = df["chose_left"][m].to_numpy()
    return float(np.mean(np.where(a_bal, y, 1 - y)))
