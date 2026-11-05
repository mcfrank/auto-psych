# name: single_example_nonmultiple_targets_rate
# description: Yes rate on one-example sets for non-example targets that are NOT multiples of the example (6->8, 14->74, 100->20, 1->27 count as non-multiples except x=1); observed above null means people generalize a lone example more diffusely than the model (below = model too diffuse beyond the multiples rule).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    d = df[n == 1]
    x = d["number_set"].astype(int)
    t = d["target"].astype(int)
    m = (t != x) & ~((x > 1) & (t % x == 0))
    if m.sum() == 0:
        return 0.0
    return float(d.loc[m, "rating"].mean())
