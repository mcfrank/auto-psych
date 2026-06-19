# name: lag1_choice_autocorrelation
# description: Mean within-participant lag-1 Pearson autocorrelation of chose_left (trials ordered by trial_index); tests for response-alternation sequential dependence that the model's trial-independence assumption cannot produce.
def test_statistic(df):
    autocorrs = []
    for pid, grp in df.groupby("participant_id"):
        y = grp.sort_values("trial_index")["chose_left"].values
        if len(y) < 3:
            continue
        r = np.corrcoef(y[:-1], y[1:])[0, 1]
        if not math.isnan(r):
            autocorrs.append(r)
    if not autocorrs:
        return float("nan")
    return float(np.mean(autocorrs))
