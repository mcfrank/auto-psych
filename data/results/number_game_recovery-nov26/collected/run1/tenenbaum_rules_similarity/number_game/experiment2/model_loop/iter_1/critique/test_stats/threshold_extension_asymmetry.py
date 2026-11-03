# name: threshold_extension_asymmetry
# description: For sets whose examples all lie in one half (all >=50 or all <=50), yes rate for out-of-range targets on the side toward the nearer end of the number line (above max for high sets, below min for low sets) minus yes rate on the opposite side; observed below null means the model over-predicts one-sided threshold extension.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float).values
    mn = s.min(axis=1).values
    mx = s.max(axis=1).values
    r = df["rating"].values.astype(float)
    hi = mn >= 50
    lo = (mx <= 50) & ~hi
    toward = (hi & (t > mx)) | (lo & (t < mn))
    away = (hi & (t < mn)) | (lo & (t > mx))
    if not toward.any() or not away.any():
        return 0.0
    return float(r[toward].mean() - r[away].mean())
