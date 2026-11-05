# name: four_minus_three_example_rate
# description: Non-example-target yes rate on four-example sets minus that on three-example sets (observed ~+0.10); observed above null means people do not narrow generalization from 3 to 4 examples as the model's accumulating size principle predicts (below = they narrow more).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    tstr = "," + df["target"].astype(str) + ","
    sets = "," + df["number_set"].str.replace(" ", "") + ","
    is_example = pd.Series([a in b for a, b in zip(tstr.values, sets.values)], index=df.index)
    r4 = df.loc[(n == 4) & ~is_example, "rating"].mean()
    r3 = df.loc[(n == 3) & ~is_example, "rating"].mean()
    v = r4 - r3
    return float(v) if np.isfinite(v) else 0.0
