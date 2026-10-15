# name: mumble_singleton_when_duplicates
# description: On mumble trials in displays containing both unique singleton objects and duplicate objects, the proportion of choices falling on unique singleton objects.
def test_statistic(df):
    sub = df[df["query"] == "prior"]
    lookup = {}
    for s in sub["objects"].unique():
        objs = json.loads(s)
        tuples = [tuple(o) for o in objs]
        counts = [tuples.count(o) for o in tuples]
        if any(c > 1 for c in counts) and any(c == 1 for c in counts):
            lookup[s] = [1 if c == 1 else 0 for c in counts]
    choices = sub["choice"].to_numpy(dtype=int)
    picks = [lookup[s][c] for s, c in zip(sub["objects"], choices) if s in lookup and c < len(lookup[s])]
    return float(np.mean(picks)) if len(picks) > 0 else 0.0
