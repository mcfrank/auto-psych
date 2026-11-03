# name: shared_final_digit_match_yes
# description: In sets of >=2 examples that all end in the same digit (excluding 0 and 5, which are multiples-of-5/10 sets), yes rate for non-example targets ending in that digit; observed below null means people use the shared-final-digit rule less than the model predicts, above means more.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    pad = np.isnan(E)
    n = (~pad).sum(axis=1)
    in_set = (E == t[:, None]).any(axis=1)
    def rate(mask):
        return float(y[mask].mean()) if mask.any() else 0.0
    d = E % 10
    same = ((d == d[:, :1]) | pad).all(axis=1) & (n >= 2) & (d[:, 0] != 0) & (d[:, 0] != 5)
    return rate(same & (t % 10 == d[:, 0]) & ~in_set)
