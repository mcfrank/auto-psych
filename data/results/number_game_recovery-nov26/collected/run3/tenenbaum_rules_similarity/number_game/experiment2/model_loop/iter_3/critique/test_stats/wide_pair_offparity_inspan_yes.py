# name: wide_pair_offparity_inspan_yes
# description: For two-example sets whose examples share parity and are >=30 apart (e.g. 13,91; 6,74), yes rate on targets strictly between the examples with the opposite parity; observed > null means people accept a broad magnitude range over the parity rule more than the model (beta width prior too restrictive), observed < null means less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    parts = sets.str.split(",", expand=True)
    if parts.shape[1] < 2:
        return 0.0
    two = parts.notna().sum(axis=1) == 2
    a = pd.to_numeric(parts[0], errors="coerce")
    b = pd.to_numeric(parts[1], errors="coerce")
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    t = df["target"].astype(int)
    m = two & (a % 2 == b % 2) & ((hi - lo) >= 30) & (t > lo) & (t < hi) & (t % 2 != a % 2)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"][m.values].astype(float).mean())
