# name: single_example_digit_share_nonmultiple
# description: Yes rate on one-example sets for non-example targets that share a digit with the example but are neither multiples nor divisors of it (14->64,74,84,94; 84->4,64); observed above null means the digit-rule discount over-suppresses digit-based generalization from a lone example (below = model still over-endorses digit neighbours).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    sub = df[n == 1]
    xs = sub["number_set"].astype(str).str.strip()
    ts = sub["target"].astype(int).astype(str)
    x = xs.astype(int); t = sub["target"].astype(int)
    share = pd.Series(False, index=sub.index)
    for d in "0123456789":
        share = share | (xs.str.contains(d, regex=False) & ts.str.contains(d, regex=False))
    arith = (t % x == 0) | (x % t == 0)
    m = share & ~arith & (t != x)
    if m.sum() == 0:
        return 0.0
    return float(sub["rating"][m].mean())
