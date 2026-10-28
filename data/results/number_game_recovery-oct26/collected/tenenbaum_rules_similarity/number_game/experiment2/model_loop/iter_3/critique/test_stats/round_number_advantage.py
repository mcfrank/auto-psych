# name: round_number_advantage
# description: Among sets with no multiple-of-10 example, yes-rate on multiple-of-10 targets minus yes-rate on targets 1-4 away from a multiple of 10 (non-examples); observed > null means people favour round numbers beyond what the model's decade-boundary interval prior produces.
def test_statistic(df):
    has10 = df["number_set"].map(lambda s: any(int(x) % 10 == 0 for x in s.split(",")))
    sub = df[~has10.to_numpy()]
    t = sub["target"].astype(int).to_numpy()
    r = sub["rating"].to_numpy().astype(float)
    exs = sub["number_set"].map(lambda s: set(int(x) for x in s.split(","))).to_numpy()
    isex = np.array([tt in e for tt, e in zip(t, exs)])
    rnd = (t % 10 == 0)
    m = t % 10
    nb = (~rnd) & (np.minimum(m, 10 - m) <= 4) & ~isex
    if rnd.sum() == 0 or nb.sum() == 0:
        return 0.0
    return float(r[rnd].mean() - r[nb].mean())
