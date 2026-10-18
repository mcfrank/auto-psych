# name: set_size_u_shape
# description: Mean yes rate on 1- and 4-example sets minus mean yes rate on 2- and 3-example sets; observed above null means generalization breadth depends non-monotonically on set size (broader for 1 and 4 examples) in a way the digit-similarity model, which has no set-size term, under-produces.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    r = df["rating"].astype(float)
    a = r[(n == 1) | (n == 4)]
    b = r[(n == 2) | (n == 3)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
