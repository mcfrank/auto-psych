# name: mumble_full_over_empty_rate
# description: On mumble trials contrasting completely empty (0 features) and completely full (all features) objects, the proportion of choices falling on the full object.
def test_statistic(df):
    sub = df[df["query"] == "prior"]
    lookup = {}
    for s in sub["objects"].unique():
        objs = json.loads(s)
        n_feats = len(objs[0])
        fcs = [sum(o) for o in objs]
        if set(fcs) == {0, n_feats}:
            lookup[s] = [1 if fc == n_feats else 0 for fc in fcs]
    choices = sub["choice"].to_numpy(dtype=int)
    picks = [lookup[s][c] for s, c in zip(sub["objects"], choices) if s in lookup]
    return float(np.mean(picks)) if len(picks) > 0 else 0.0
