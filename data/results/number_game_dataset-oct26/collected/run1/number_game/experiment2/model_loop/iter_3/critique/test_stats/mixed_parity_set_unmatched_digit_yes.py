# name: mixed_parity_set_unmatched_digit_yes
# description: In sets containing both odd and even examples, yes rate for targets whose final digit matches no example (not examples); t_observed above null_mean means people treat parity-mixed sets as a broad "any number" category more than the model's averaged parity credit predicts; below means the model over-generalizes in mixed sets.
def test_statistic(df):
    info = {}
    for s in df["number_set"].astype(str).unique():
        xs = [int(x) for x in s.split(",")]
        info[s] = (len(set(x % 2 for x in xs)) == 2, set(x % 10 for x in xs), set(xs))
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    m = np.array([info[s][0] and (v % 10 not in info[s][1]) and (v not in info[s][2]) for s, v in zip(sets, t)])
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).values[m].mean())
