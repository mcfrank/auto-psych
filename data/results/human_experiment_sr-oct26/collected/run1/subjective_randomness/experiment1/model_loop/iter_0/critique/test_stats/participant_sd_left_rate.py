# name: participant_sd_left_rate
# description: Standard deviation across participants of their overall chose_left rate; observed > null means participants carry individual side biases/response styles that the model's single shared side_bias underproduces.
def test_statistic(df):
    return float(df.groupby("participant_id")["chose_left"].mean().std(ddof=0))
