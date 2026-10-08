# name: longest_run_avoidance_equal_switches
# description: Among pairs whose switch counts differ by at most 1 but whose longest runs differ, the proportion of choices for the sequence with the shorter longest run; observed above null_mean means people penalise long streaks beyond what switch/motif statistics explain (model under-penalises streaks), below means over-penalises.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    def switches(s):
        return s.str.count(r"(?=HT)|(?=TH)")
    def longest(s):
        return s.str.findall(r"H+|T+").map(lambda L: max(len(x) for x in L))
    ka, kb = switches(a), switches(b)
    la, lb = longest(a), longest(b)
    m = ((ka - kb).abs() <= 1) & (la != lb)
    if m.sum() == 0:
        return 0.5
    a_short = (la < lb)[m].to_numpy()
    y = df["chose_left"][m].to_numpy()
    return float(np.mean(np.where(a_short, y, 1 - y)))
