# name: periodicity_b_choice_correlation
# description: Pearson correlation between periodicity_b and chose_left across all trials; tests whether a feature entirely absent from the model (periodicity of sequence B) independently predicts human choice direction.
def test_statistic(df):
    r = np.corrcoef(df["periodicity_b"].values, df["chose_left"].values)[0, 1]
    return float(r)
