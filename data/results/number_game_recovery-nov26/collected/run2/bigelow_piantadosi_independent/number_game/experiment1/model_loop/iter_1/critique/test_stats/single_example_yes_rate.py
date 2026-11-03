# name: single_example_yes_rate
# description: Yes rate on trials whose set has exactly one example; observed > null means the model generalizes too narrowly from a single example (size principle too strong), observed < null means too broadly.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",").to_numpy() + 1
    m = n == 1
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
