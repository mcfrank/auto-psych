# name: target_last_minus_first_example_yes
# description: In multi-example sets, yes rate when the target equals the last-shown example minus yes rate when it equals the first-shown example; observed above null means people endorse the last example more (recency) than the model's last-example weight implies, below null means they favour the first.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",")
    n = parts.str.len()
    first = parts.str[0].astype(int)
    last = parts.str[-1].astype(int)
    t = df["target"].astype(int)
    y = df["rating"].astype(float)
    m = n > 1
    a = y[m & (t == last)]
    b = y[m & (t == first)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
