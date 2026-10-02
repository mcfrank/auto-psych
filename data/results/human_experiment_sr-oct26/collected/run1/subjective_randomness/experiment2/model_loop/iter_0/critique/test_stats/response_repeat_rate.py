# name: response_repeat_rate
# description: Proportion of consecutive trials (within participant, by trial_index) on which the same side (Left/Right) was chosen as on the previous trial; observed > null means response perseveration the model underproduces, observed < null means excess side switching.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    same_p = d["participant_id"].values[1:] == d["participant_id"].values[:-1]
    c = d["chose_left"].values
    rep = c[1:] == c[:-1]
    return float(rep[same_p].mean())
