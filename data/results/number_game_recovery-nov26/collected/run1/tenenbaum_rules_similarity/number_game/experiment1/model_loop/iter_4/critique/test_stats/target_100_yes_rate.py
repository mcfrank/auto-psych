# name: target_100_yes_rate
# description: Yes-rate on trials whose target is 100 and 100 is not one of the examples; observed < null means the model over-produces "yes" to the endpoint 100 (people show an extreme-number aversion that parity/multiples rules containing 100 do not capture).
def test_statistic(df):
    t = df["target"].astype(int)
    ex = df["number_set"].astype(str)
    in_set = ex.str.split(",").apply(lambda l: "100" in [x.strip() for x in l])
    m = (t == 100) & ~in_set
    return float(df.loc[m, "rating"].mean()) if m.any() else 0.0
