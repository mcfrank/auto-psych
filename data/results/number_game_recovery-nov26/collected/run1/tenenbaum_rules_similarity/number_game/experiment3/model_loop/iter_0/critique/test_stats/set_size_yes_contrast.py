# name: set_size_yes_contrast
# description: Overall yes rate on 4-example sets minus yes rate on 2-example sets; observed more negative than null means people narrow their generalisation with more examples more strongly than the model's size principle predicts, more positive means the model narrows too fast.
def test_statistic(df):
    n = (df["number_set"].astype(str).str.count(",") + 1).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    a = y[n == 4]; b = y[n == 2]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
