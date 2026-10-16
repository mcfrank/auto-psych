# name: heads_count_preference
# description: Among trials whose sequences have different numbers of H, the proportion choosing the sequence with more heads; the model is H/T symmetric, so observed > null means people favour head-heavy sequences as random, observed < null that they favour tail-heavy ones (a label asymmetry the model cannot produce).
def test_statistic(df):
    ha = df["sequence_a"].str.count("H").to_numpy()
    hb = df["sequence_b"].str.count("H").to_numpy()
    m = ha != hb
    y = df["chose_left"].to_numpy()
    return float(np.where(ha > hb, y, 1 - y)[m].mean())
