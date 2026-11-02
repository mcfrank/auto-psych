# name: participant_yes_rate_sd
# description: Standard deviation across participants (a new participant starts whenever trial_index resets downward) of the overall yes-rate; observed > null means people differ in response bias/generalisation breadth more than the model's single shared parameter set implies.
def test_statistic(df):
    ti = df["trial_index"].astype(int)
    pid = (ti.diff() < 0).cumsum()
    rates = df["rating"].astype(float).groupby(pid).mean()
    return float(rates.std()) if len(rates) > 1 else 0.0
