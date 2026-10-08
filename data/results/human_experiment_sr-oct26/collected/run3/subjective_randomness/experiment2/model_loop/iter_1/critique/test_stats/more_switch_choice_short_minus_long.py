# name: more_switch_choice_short_minus_long
# description: Proportion choosing the sequence with more switches at lengths <= 3 minus the same at length 8 (trials with unequal switch counts); observed != null means the model's length scaling misstates how switch preference changes with sequence length (positive: stronger alternation preference in short sequences).
def _sw(s):
    return sum(1 for x, y in zip(s, s[1:]) if x != y)
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    u = pd.unique(pd.concat([a, b])); sw = {s: _sw(s) for s in u}
    ka = a.map(sw).to_numpy(); kb = b.map(sw).to_numpy()
    n = a.str.len().to_numpy(); y = df["chose_left"].to_numpy()
    more = np.where(ka > kb, y, 1 - y).astype(float)
    d = ka != kb
    s = d & (n <= 3); l = d & (n == 8)
    if s.sum() == 0 or l.sum() == 0:
        return 0.0
    return float(more[s].mean() - more[l].mean())
