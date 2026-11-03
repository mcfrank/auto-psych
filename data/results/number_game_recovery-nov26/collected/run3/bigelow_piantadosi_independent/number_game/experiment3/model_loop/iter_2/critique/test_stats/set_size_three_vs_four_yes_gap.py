# name: set_size_three_vs_four_yes_gap
# description: Overall yes rate on 3-example sets minus yes rate on 4-example sets; observed above null means people generalise more after three examples (and tighten more at four) than the model's set-size evidence weighting predicts, below null means the model overstates the 3-vs-4 difference.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    r = df["rating"].astype(float)
    a = r[n == 3]
    b = r[n == 4]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
