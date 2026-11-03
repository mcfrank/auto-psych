# name: participant_yes_rate_sd
# description: Standard deviation across participants (sessions split where trial_index resets) of the overall yes rate; observed > null means individuals differ in yes-bias more than the single-parameter-set model allows.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[0, (np.diff(ti) < 0).astype(int)])
    y = df["rating"].astype(float).to_numpy()
    rates = pd.Series(y).groupby(pid).mean().to_numpy()
    return float(rates.std()) if len(rates) > 1 else 0.0
