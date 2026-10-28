# name: singleton_set_yes_rate
# description: Yes-rate on trials whose set has a single example; observed above null means the model generalizes too narrowly from one example (too strong a size principle / rule preference), below means too broadly.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",").to_numpy() + 1
    r = df["rating"].astype(float).to_numpy()
    sel = n == 1
    return float(r[sel].mean()) if sel.any() else 0.0
