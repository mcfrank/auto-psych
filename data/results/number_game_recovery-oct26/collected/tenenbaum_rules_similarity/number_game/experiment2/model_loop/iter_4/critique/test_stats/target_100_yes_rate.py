# name: target_100_yes_rate
# description: Yes-rate when the target is 100 and 100 is not among the examples; observed below null means the model over-credits ranges/rules reaching the domain's top (e.g. decade-aligned ranges ending at 100), above null means people treat 100 as more salient than the model.
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    s = ("," + df["number_set"].astype(str).str.replace(" ", "") + ",")
    has100 = s.str.contains(",100,").to_numpy()
    m = (t == 100) & ~has100
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
