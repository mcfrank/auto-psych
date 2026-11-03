# name: near_example_yes_rate
# description: In sets with >=3 examples, the "yes" rate for targets 1-2 away from the nearest example (not in the set) minus the rate for targets 3-10 away; observed above null = people show a sharper local-similarity gradient than the model's range/rule mixture produces.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    t = df["target"].astype(int)
    dist = ex.sub(t, axis=0).abs().min(axis=1)
    near = (n >= 3) & (dist >= 1) & (dist <= 2)
    mid = (n >= 3) & (dist >= 3) & (dist <= 10)
    if near.sum() == 0 or mid.sum() == 0:
        return 0.0
    return float(df.loc[near, "rating"].mean() - df.loc[mid, "rating"].mean())
