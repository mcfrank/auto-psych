# name: multi_zero_consensus_fraction
# description: Fraction of (set, target) pairs from multi-example sets where none of the 9 answers is yes; observed > null means people give unanimous "no" more often than the model, whose doubt-mode flat yes puts a floor under every target (observed < null: floor too low).
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",").values + 1
    d = df.loc[n >= 2, ["number_set", "target", "rating"]]
    s = d.groupby(["number_set", "target"])["rating"].sum()
    return float((s == 0).mean()) if len(s) else 0.0
