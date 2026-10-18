# name: single_example_same_decade_other_digit_yes
# description: In single-example sets, yes rate for targets in the same decade (t//10 == x//10) as the example but with a different final digit; t_observed above null_mean means people generalize within a decade more than the model (digit similarity + band) predicts; below means the model over-produces same-decade yeses.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    x = pd.to_numeric(s.where(single, "0"), errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    m = single.values & (t // 10 == x // 10) & (t % 10 != x % 10)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).values[m].mean())
