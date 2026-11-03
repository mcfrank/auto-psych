# name: person_session_drift_sd
# description: SD across participants of (yes rate in second half of session minus first half, by trial_index); observed above null means individuals drift in yes-propensity over the session in different directions, which the model's constant per-person parameters cannot produce.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(ti == 0)
    r = df["rating"].to_numpy().astype(float)
    d = pd.DataFrame({"pid": pid, "late": ti >= 225, "r": r})
    m = d.groupby(["pid", "late"])["r"].mean().unstack()
    diff = (m[True] - m[False]).dropna()
    if len(diff) < 2:
        return 0.0
    return float(diff.std(ddof=1))
