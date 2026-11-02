# name: set_size_yes_contrast
# description: Overall yes-rate for 4-example sets minus that for 1-2-example sets; observed > null means people tighten generalisation with more examples less than the model's size principle predicts, observed < null means they tighten more.
def test_statistic(df):
    n = df["number_set"].str.count(",").to_numpy() + 1
    r = df["rating"].to_numpy(float)
    a = n >= 4; b = n <= 2
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
