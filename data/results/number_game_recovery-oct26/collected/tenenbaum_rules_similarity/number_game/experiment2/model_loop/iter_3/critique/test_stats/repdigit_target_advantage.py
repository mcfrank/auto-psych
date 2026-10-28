# name: repdigit_target_advantage
# description: Among sets containing no repdigit example (11,22,...,99), yes-rate on repdigit targets minus yes-rate on non-repdigit targets within +-3 of a repdigit; observed > null means people treat "same-digit" numbers as a salient concept the model lacks (observed < null: the model over-generalizes to them).
def test_statistic(df):
    rep = {11, 22, 33, 44, 55, 66, 77, 88, 99}
    has_rep = df["number_set"].map(lambda s: any(int(x) in rep for x in s.split(",")))
    sub = df[~has_rep.to_numpy()]
    t = sub["target"].astype(int).to_numpy()
    r = sub["rating"].to_numpy().astype(float)
    isrep = np.isin(t, list(rep))
    d = np.abs((t[:, None] - np.array(sorted(rep))[None, :])).min(axis=1)
    nb = (~isrep) & (d <= 3) & (t >= 10)
    if isrep.sum() == 0 or nb.sum() == 0:
        return 0.0
    return float(r[isrep].mean() - r[nb].mean())
