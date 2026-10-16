# name: balance_preference_equal_switches
# description: Among trials whose sequences have equal switch counts but different |#H - #T|, the proportion choosing the more H/T-balanced sequence; observed > null means the model under-produces preference for balanced head/tail counts.
def _sw(s):
    return sum(1 for x, y in zip(s, s[1:]) if x != y)
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    n = a.str.len().to_numpy()
    ia = np.abs(2 * a.str.count("H").to_numpy() - n); ib = np.abs(2 * b.str.count("H").to_numpy() - n)
    u = pd.unique(pd.concat([a, b])); sw = {s: _sw(s) for s in u}
    ka = a.map(sw).to_numpy(); kb = b.map(sw).to_numpy()
    m = (ka == kb) & (ia != ib)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    return float(np.where(ia[m] < ib[m], y, 1 - y).mean())
