# name: single_example_digit_vs_adjacent
# description: For single-example sets, yes rate for distant (>10 away) targets sharing the example's final digit minus yes rate for adjacent (1-3 away) targets with a different final digit; observed > model means people weight a shared final digit over numeric closeness more than the model's baseline attention allows.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    x = pd.to_numeric(s.where(single, "0")).astype(int).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    d = np.abs(t - x)
    same = (t % 10) == (x % 10)
    sm = single.values
    a = sm & same & (d > 10)
    b = sm & (~same) & (d >= 1) & (d <= 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
