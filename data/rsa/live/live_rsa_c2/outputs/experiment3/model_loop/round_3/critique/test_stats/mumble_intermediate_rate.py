# name: mumble_intermediate_rate
# description: Proportion of choices on mumble trials falling on strictly intermediate-complexity objects whose feature count is strictly between the display minimum and maximum.
def test_statistic(df):
    sub = df[df["query"] == "prior"]
    lookup = {}
    for s in sub["objects"].unique():
        objs = json.loads(s)
        fcs = [sum(o) for o in objs]
        min_f, max_f = min(fcs), max(fcs)
        lookup[s] = [1 if (fc > min_f and fc < max_f) else 0 for fc in fcs]
    choices = sub["choice"].to_numpy(dtype=int)
    picks = [lookup[s][c] for s, c in zip(sub["objects"], choices)]
    return float(np.mean(picks)) if len(picks) > 0 else 0.0
