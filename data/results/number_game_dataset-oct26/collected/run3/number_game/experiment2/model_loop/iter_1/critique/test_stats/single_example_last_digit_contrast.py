# name: single_example_last_digit_contrast
# description: For single-example sets, yes-rate on (non-example) targets sharing the example's last digit minus yes-rate on other non-example targets; observed > model means people lean on the "same last digit" rule after one example more than the model's guessing-diluted posterior predicts, observed < model means less.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    e = pd.to_numeric(s.where(single, "-1"), errors="coerce").fillna(-1).astype(int).to_numpy()
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].to_numpy(dtype=float)
    base = single.to_numpy() & (t != e)
    same = base & (t % 10 == e % 10)
    other = base & (t % 10 != e % 10)
    if same.sum() == 0 or other.sum() == 0:
        return 0.0
    return float(y[same].mean() - y[other].mean())
