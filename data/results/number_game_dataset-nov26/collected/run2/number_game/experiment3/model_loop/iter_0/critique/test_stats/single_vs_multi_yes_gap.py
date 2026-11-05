# name: single_vs_multi_yes_gap
# description: Yes rate on one-example sets minus yes rate on 2-4-example sets; observed > null means the model under-produces the extra breadth of generalization people show from a single example (or over-produces it if observed < null).
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    r = df["rating"].astype(float)
    a = r[n == 1].mean(); b = r[n >= 2].mean()
    return float(a - b)
