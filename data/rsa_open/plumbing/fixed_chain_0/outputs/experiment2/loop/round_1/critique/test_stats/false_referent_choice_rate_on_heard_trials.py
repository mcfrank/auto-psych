# name: false_referent_choice_rate_on_heard_trials
# description: Rate of choosing an object for which the heard word is literally false (`objects[choice][heard] == 0`) on heard-word trials where at least one object in the scene lacks the heard feature.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    false_map = {}
    for objs_str, heard in zip(scenes["objects"], scenes["heard"]):
        if pd.isna(heard):
            continue
        h = int(heard)
        objs = json.loads(objs_str)
        false_idxs = [i for i, o in enumerate(objs) if o[h] == 0]
        if false_idxs:
            false_map[(objs_str, h)] = set(false_idxs)
    if not false_map:
        return 0.0
    valid = df["heard"].notna()
    sub = df[valid]
    keys = list(zip(sub["objects"], sub["heard"].astype(int)))
    targets = [false_map.get(k) for k in keys]
    mask = [t is not None for t in targets]
    if not any(mask):
        return 0.0
    choices = sub["choice"].to_numpy()[mask]
    active_targets = [t for t in targets if t is not None]
    hits = [1.0 if c in t else 0.0 for c, t in zip(choices, active_targets)]
    return float(np.mean(hits))
