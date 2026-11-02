# name: participant_drift_sd
# description: SD across participants (split at trial_index==0) of the change in yes rate from the first to the second half of their session; above the null means individuals drift in their yes-bias over time (some up, some down) more than a static model's sampling noise allows.
def test_statistic(df):
    pid = np.cumsum(df["trial_index"].astype(int).values == 0)
    half = df["trial_index"].astype(int).values >= 225
    y = df["rating"].astype(float).values
    d = pd.DataFrame({"p": pid, "h": half, "y": y})
    m = d.groupby(["p", "h"]).y.mean().unstack()
    m = m.dropna()
    if len(m) < 2:
        return 0.0
    return float((m[True] - m[False]).std())
