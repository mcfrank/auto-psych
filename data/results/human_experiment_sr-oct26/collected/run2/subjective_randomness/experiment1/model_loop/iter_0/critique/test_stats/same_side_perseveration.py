# name: same_side_perseveration
# description: Rate at which a participant chooses the same screen side (left/right) as on their previous trial; observed > null means response-side perseveration the trial-independent model does not produce, observed < null means excess side switching.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    y = d["chose_left"].to_numpy(); p = d["participant_id"].to_numpy()
    same_p = p[1:] == p[:-1]
    if same_p.sum() == 0:
        return 0.5
    return float((y[1:] == y[:-1])[same_p].mean())
