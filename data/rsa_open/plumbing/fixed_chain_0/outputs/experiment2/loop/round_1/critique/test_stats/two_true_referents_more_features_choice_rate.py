# name: two_true_referents_more_features_choice_rate
# description: Rate of choosing the more feature-rich word-true referent on heard-word trials where the heard word is true of exactly two objects and those two objects differ in their total number of features.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    target_map = {}
    for objs_str, heard in zip(scenes["objects"], scenes["heard"]):
        if pd.isna(heard):
            continue
        h = int(heard)
        objs = json.loads(objs_str)
        true_idxs = [i for i, o in enumerate(objs) if o[h] == 1]
        if len(true_idxs) == 2:
            fc0, fc1 = sum(objs[true_idxs[0]]), sum(objs[true_idxs[1]])
            if fc0 != fc1:
                rich_idx = true_idxs[0] if fc0 > fc1 else true_idxs[1]
                target_map[(objs_str, h)] = rich_idx
    if not target_map:
        return 0.0
    valid = df["heard"].notna()
    sub = df[valid]
    keys = list(zip(sub["objects"], sub["heard"].astype(int)))
    targets = [target_map.get(k) for k in keys]
    mask = [t is not None for t in targets]
    if not any(mask):
        return 0.0
    choices = sub["choice"].to_numpy()[mask]
    active_targets = np.array([t for t in targets if t is not None])
    return float(np.mean(choices == active_targets))
