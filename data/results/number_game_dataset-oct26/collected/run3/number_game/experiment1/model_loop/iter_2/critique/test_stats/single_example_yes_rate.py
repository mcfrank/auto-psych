# name: single_example_yes_rate
# description: Yes-rate on trials whose set has exactly one example (target different from it); observed above null means the model still generalizes too narrowly from a single example, below means too broadly.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    one = ~ns.str.contains(",")
    ok = one & (pd.to_numeric(ns, errors="coerce") != df["target"].astype(int))
    if not ok.any():
        return 0.5
    return float(df["rating"][ok].astype(float).mean())
