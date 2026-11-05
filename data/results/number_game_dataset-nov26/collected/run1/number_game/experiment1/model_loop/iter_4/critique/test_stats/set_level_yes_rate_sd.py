# name: set_level_yes_rate_sd
# description: SD across sets (with >=3 distinct targets) of each set's mean yes-rate; observed > null means people's overall willingness to say yes varies by set more than the model's single similarity-to-response mapping produces (set-specific rule breadth), observed < null means less.
def test_statistic(df):
    g = df.groupby("number_set")
    m = g["rating"].mean()
    nt = g["target"].nunique()
    v = m[nt >= 3].values.astype(float)
    return float(np.std(v)) if len(v) > 1 else 0.0
