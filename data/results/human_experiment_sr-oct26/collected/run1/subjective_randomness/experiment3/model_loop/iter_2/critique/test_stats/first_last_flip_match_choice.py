# name: first_last_flip_match_choice
# description: Among trials where one sequence starts and ends with the same flip and the other does not, the proportion choosing the one whose first and last flips MATCH; observed above null_mean means people find matching endpoints more random than the model (which has no endpoint-identity term) predicts, below means they find them less random.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ma = (a.str[0] == a.str[-1]).to_numpy()
    mb = (b.str[0] == b.str[-1]).to_numpy()
    mask = ma != mb
    if not mask.any():
        return 0.5
    c = df["chose_left"].to_numpy()
    chose_match = np.where(ma, c, 1 - c)
    return float(chose_match[mask].mean())
