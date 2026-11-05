# name: shared_digit_matching_outside_range
# description: For 3+ example sets whose examples all share a last digit, yes rate on non-example targets with that last digit lying outside the examples' range; observed below null means the model over-predicts (near-ceiling) endorsement of rule-matching targets outside the range.
def test_statistic(df):
    s = df["number_set"].astype(str)
    sp = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = sp.notna().sum(axis=1).to_numpy()
    ld = sp % 10
    same = (ld.max(axis=1) == ld.min(axis=1)).to_numpy()
    d = ld.max(axis=1).to_numpy()
    lo = sp.min(axis=1).to_numpy(); hi = sp.max(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    isex = (sp.to_numpy() == t[:, None]).any(axis=1)
    m = (n >= 3) & same & (t % 10 == d) & ((t < lo) | (t > hi)) & ~isex
    return float(r[m].mean()) if m.sum() > 0 else 0.0
