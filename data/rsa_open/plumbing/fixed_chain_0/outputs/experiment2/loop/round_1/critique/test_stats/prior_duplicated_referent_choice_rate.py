# name: prior_duplicated_referent_choice_rate
# description: Rate of choosing a duplicated (non-unique) object on wordless prior trials (`heard` is NaN) in scenes that contain both unique (singleton) and duplicated objects.
def test_statistic(df):
    scenes = df[df["heard"].isna()][["objects"]].drop_duplicates()
    target_map = {}
    for objs_str in scenes["objects"]:
        objs = json.loads(objs_str)
        dups = [sum(1 for o2 in objs if o2 == objs[i]) for i in range(len(objs))]
        dup_idxs = [i for i, d in enumerate(dups) if d > 1]
        sing_idxs = [i for i, d in enumerate(dups) if d == 1]
        if dup_idxs and sing_idxs:
            target_map[objs_str] = set(dup_idxs)
    if not target_map:
        return 0.0
    sub = df[df["heard"].isna()]
    targets = [target_map.get(o) for o in sub["objects"]]
    mask = [t is not None for t in targets]
    if not any(mask):
        return 0.0
    choices = sub["choice"].to_numpy()[mask]
    active_targets = [t for t in targets if t is not None]
    hits = [1.0 if c in t else 0.0 for c, t in zip(choices, active_targets)]
    return float(np.mean(hits))
