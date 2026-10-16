# name: switch_choice_sequential_dependence
# description: Over consecutive trial pairs (within participant, by trial_index) where both trials' sequences differ in switch count, the proportion where the person makes the same kind of choice (picks the more-switching sequence on both or the fewer-switching on both); observed > null means choices carry over a criterion from the previous trial more than the model's static person parameters produce, < null that people deliberately vary their criterion.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    a = d["sequence_a"]; b = d["sequence_b"]
    n = a.str.len().max()
    A = np.array([list(s.ljust(n, "_")) for s in a])
    B = np.array([list(s.ljust(n, "_")) for s in b])
    ka = ((A[:, 1:] != A[:, :-1]) & (A[:, 1:] != "_")).sum(1)
    kb = ((B[:, 1:] != B[:, :-1]) & (B[:, 1:] != "_")).sum(1)
    y = d["chose_left"].to_numpy()
    valid = ka != kb
    more = np.where(ka > kb, y, 1 - y)
    p = d["participant_id"].to_numpy()
    m = (p[1:] == p[:-1]) & valid[1:] & valid[:-1]
    return float((more[1:] == more[:-1])[m].mean())
