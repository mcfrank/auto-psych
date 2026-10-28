# name: prior_featureless_singleton_choice_rate
# description: Rate of choosing the completely featureless object (`[0, ..., 0]`) on wordless prior trials (`heard` is NaN) where the featureless object is a unique singleton and at least one other singleton object with features is present.
def test_statistic(df):
    scenes = df[df["heard"].isna()][["objects"]].drop_duplicates()
    target_map = {}
    for objs_str in scenes["objects"]:
        objs = json.loads(objs_str)
        dups = [sum(1 for o2 in objs if o2 == objs[i]) for i in range(len(objs))]
        fcs = [sum(o) for o in objs]
        zero_sings = [i for i in range(len(objs)) if fcs[i] == 0 and dups[i] == 1]
        nonzero_sings = [i for i in range(len(objs)) if fcs[i] > 0 and dups[i] == 1]
        if zero_sings and nonzero_sings:
            target_map[objs_str] = zero_sings[0]
    if not target_map:
        return 0.0
    sub = df[df["heard"].isna()]
    targets = [target_map.get(o) for o in sub["objects"]]
    mask = [t is not None for t in targets]
    if not any(mask):
        return 0.0
    choices = sub["choice"].to_numpy()[mask]
    active_targets = np.array([t for t in targets if t is not None])
    return float(np.mean(choices == active_targets))
