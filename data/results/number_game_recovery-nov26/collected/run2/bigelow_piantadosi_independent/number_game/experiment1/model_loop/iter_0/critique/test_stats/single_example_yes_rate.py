# name: single_example_yes_rate
# description: Yes rate among trials whose set has exactly one example (target not equal to it); t_observed above null_mean means people generalize more broadly from one example than the model does.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    first = df["number_set"].astype(str).str.split(",").str[0].astype(int)
    m = (n == 1) & (first != df["target"].astype(int))
    if m.sum() == 0:
        return 0.0
    return float(df["rating"][m].mean())
