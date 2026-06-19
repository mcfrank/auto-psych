# name: na4_choice_rate
# description: Mean chose_left restricted to trials where n_a = 4 (medium-length A sequences with low alternation); tests whether the model correctly predicts near-0 left-choice for these streaky A sequences that the Bayesian score may mis-score as fair-coin diagnostic.
def test_statistic(df):
    sub = df[df["n_a"] == 4]
    if len(sub) == 0:
        return float("nan")
    return float(sub["chose_left"].mean())
