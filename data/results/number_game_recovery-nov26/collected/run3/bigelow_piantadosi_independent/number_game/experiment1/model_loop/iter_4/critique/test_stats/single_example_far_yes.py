# name: single_example_far_yes
# description: For single-example sets, yes rate on targets more than 10 away from the example; observed > null means people generalise from one number to distant numbers (via rules) more than the model, observed < null means the model over-generalises from a single example.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    e = pd.to_numeric(s.where(single), errors="coerce")
    t = df["target"].astype(float)
    sel = single & ((t - e).abs() > 10)
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float)[sel].mean())
