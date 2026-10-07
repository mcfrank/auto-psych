# name: starts_with_heads_preference
# description: Among trials whose two sequences begin with different flips, the proportion choosing the sequence that starts with H; the model is H/T symmetric, so observed > null means people see H-initial sequences as more random (a start-label asymmetry the model cannot produce), < null T-initial ones.
def test_statistic(df):
    fa = df["sequence_a"].str[0].to_numpy()
    fb = df["sequence_b"].str[0].to_numpy()
    m = fa != fb
    y = df["chose_left"].to_numpy()
    return float(np.where(fa == "H", y, 1 - y)[m].mean())
