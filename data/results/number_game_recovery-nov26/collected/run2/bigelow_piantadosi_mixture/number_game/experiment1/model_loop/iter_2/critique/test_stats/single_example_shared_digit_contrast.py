# name: single_example_shared_digit_contrast
# description: For one-example sets (target != example), yes rate for targets sharing a digit with the example minus yes rate for targets sharing none; observed above the null means the model under-produces digit-based generalization from a single example (equal prior underweights digit rules), below means it over-produces it.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    x = pd.to_numeric(s.where(single, "0")).astype(int).to_numpy()
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    td = np.array([[str(d) in str(v) for d in range(10)] for v in range(101)])
    share = np.any(td[x] & td[t], axis=1)
    m = single.to_numpy() & (t != x)
    a = y[m & share]; b = y[m & ~share]
    return float((a.mean() if a.size else 0.0) - (b.mean() if b.size else 0.0))
