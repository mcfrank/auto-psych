# name: lag2_irregular_equal_switches
# description: Among pairs of length >= 5 with equal switch counts but different lag-2 mismatch counts (positions i where flip i != flip i-2), the rate of choosing the sequence with more lag-2 mismatches (less period-2 regularity); observed above null means people penalise period-2 / motif repetition more than the model's motif generator predicts, below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    uniq = pd.unique(pd.concat([a, b]).values)
    l2 = {s: sum(1 for i in range(2, len(s)) if s[i] != s[i - 2]) for s in uniq}
    ma = a.map(l2).values; mb = b.map(l2).values
    sa = (a.str.count("HT") + a.str.count("TH")).values; sb = (b.str.count("HT") + b.str.count("TH")).values
    n = a.str.len().values
    m = (n >= 5) & (sa == sb) & (ma != mb)
    if m.sum() == 0:
        return 0.5
    chose_irr = np.where(ma > mb, df["chose_left"].values, 1 - df["chose_left"].values)
    return float(chose_irr[m].mean())
