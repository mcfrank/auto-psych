# name: single_example_nonmultiples_rate
# description: Yes rate on one-example sets for targets that are NOT multiples of the lone example and not the example itself; observed above null means the model generalizes too narrowly (too little diffuse, similarity-like endorsement) from a single example.
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    s = df[n == 1]
    x = s["number_set"].astype(int)
    t = s["target"].astype(int)
    m = (t % x != 0)
    if m.sum() == 0:
        return 0.0
    return float(s.loc[m, "rating"].mean())
