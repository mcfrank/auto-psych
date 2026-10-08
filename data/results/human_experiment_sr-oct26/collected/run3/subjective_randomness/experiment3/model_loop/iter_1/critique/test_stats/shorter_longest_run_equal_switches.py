# name: shorter_longest_run_equal_switches
# description: Among pairs with equal switch counts but different longest-run lengths, the rate of choosing the sequence whose longest run is shorter; observed above null means people penalise a single long streak more than the model (given switch count) predicts, below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    uniq = pd.unique(pd.concat([a, b]).values)
    lr = {s: max(len(r) for r in s.replace("HT", "H T").replace("TH", "T H").split()) for s in uniq}
    la = a.map(lr).values; lb = b.map(lr).values
    sa = (a.str.count("HT") + a.str.count("TH")).values; sb = (b.str.count("HT") + b.str.count("TH")).values
    m = (sa == sb) & (la != lb)
    if m.sum() == 0:
        return 0.5
    chose_shorter = np.where(la < lb, df["chose_left"].values, 1 - df["chose_left"].values)
    return float(chose_shorter[m].mean())
