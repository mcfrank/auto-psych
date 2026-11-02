# name: single_square_example_square_target_yes
# description: Yes rate for perfect-square targets (other than the example itself) on single-example sets whose example is a perfect square (4, 9, ..., 100); observed above null means people read a lone square as "squares" more readily than the model's equal familiar-rule prior allows (model under-produces), below means the model over-endorses the squares rule.
def test_statistic(df):
    sq = {k * k for k in range(1, 11)}
    ns = df["number_set"].astype(str)
    single = ~ns.str.contains(",")
    ex = pd.to_numeric(ns.where(single, "0"), errors="coerce").fillna(0).astype(int)
    t = df["target"].astype(int)
    m = single & ex.isin(sq) & t.isin(sq) & (t != ex)
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].astype(float).mean())
