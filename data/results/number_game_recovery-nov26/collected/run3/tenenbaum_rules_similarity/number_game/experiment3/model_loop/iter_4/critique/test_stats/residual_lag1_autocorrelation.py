# name: residual_lag1_autocorrelation
# description: Within-participant correlation between consecutive answers' residuals (answer minus its (set,target) cell mean yes rate), participants split at trial_index==0; above the null means answers carry over from one trial to the next beyond what the stimuli explain (response perseveration/momentary bias the model lacks).
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    y = df["rating"].astype(float)
    res = (y - y.groupby(key).transform("mean")).values
    pid = np.cumsum(df["trial_index"].astype(int).values == 0)
    same = pid[1:] == pid[:-1]
    a, b = res[1:][same], res[:-1][same]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
