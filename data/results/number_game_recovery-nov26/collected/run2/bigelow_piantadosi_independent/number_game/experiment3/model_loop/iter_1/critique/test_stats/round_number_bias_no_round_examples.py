# name: round_number_bias_no_round_examples
# description: For sets containing no multiple of 10, yes-rate on multiple-of-10 targets minus yes-rate on other non-example targets; observed above null_mean means people over-endorse round numbers as a salience bias the rule/range model under-produces.
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    u = df["number_set"].unique()
    noround = {s: all(int(x) % 10 for x in s.split(",")) for s in u}
    m = df["number_set"].map(noround).to_numpy().astype(bool)
    isex = np.array([str(tt) in s.split(",") for s, tt in zip(df["number_set"].to_numpy(), t)])
    r = df["rating"].astype(float).to_numpy()
    a = m & ~isex & (t % 10 == 0); b = m & ~isex & (t % 10 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
