# name: participant_left_rate_sd
# description: Standard deviation across participants of each participant's Left-choice rate; t_observed > null_mean means individual side biases (or response noise heterogeneity) the model lacks.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().std(ddof=0))
