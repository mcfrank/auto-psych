# name: within_trial_choice_variance
# description: Mean of the within-trial sample variance of chose_left (computed as k*(n-k)/(n*(n-1)) per trial where k = sum chose_left and n = participants per trial), averaged across the 32 unique trials; detects overdispersion relative to binomial predictions from individual differences the model cannot represent.
def test_statistic(df):
    variances = []
    for _, grp in df.groupby("trial_index"):
        k = grp["chose_left"].sum()
        n = len(grp)
        if n < 2:
            continue
        v = k * (n - k) / (n * (n - 1))
        variances.append(v)
    if not variances:
        return float("nan")
    return float(np.mean(variances))
