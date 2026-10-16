# name: response_perseveration
# description: Proportion of consecutive trials (within participant, ordered by trial_index) on which the participant pressed the same button (Left/Right) as on the previous trial; observed above null means people perseverate on the button more than the model (which has no sequential dependence), below means they alternate buttons more.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    y = d["chose_left"].to_numpy()
    p = d["participant_id"].to_numpy()
    same_p = p[1:] == p[:-1]
    rep = (y[1:] == y[:-1])[same_p]
    return float(rep.mean())
