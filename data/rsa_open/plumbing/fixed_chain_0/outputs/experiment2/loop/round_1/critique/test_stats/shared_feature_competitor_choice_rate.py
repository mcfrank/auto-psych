# name: shared_feature_competitor_choice_rate
# description: Rate of choosing a non-minimal word-true referent on heard-word trials where a single-feature word-true referent competes against multi-feature word-true referents that share additional non-heard features with other objects in the display.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    target_map = {}
    for objs_str, heard in zip(scenes["objects"], scenes["heard"]):
        if pd.isna(heard):
            continue
        h = int(heard)
        objs = json.loads(objs_str)
        n_feat = len(objs[0])
        true_idxs = [i for i, o in enumerate(objs) if o[h] == 1]
        dups = [sum(1 for o2 in objs if o2 == objs[i]) for i in range(len(objs))]
        fcs = [sum(o) for o in objs]
        single_feat_sing = [i for i in true_idxs if fcs[i] == 1 and dups[i] == 1]
        if not single_feat_sing:
            continue
        # Check for multi-feature word-true referents whose non-heard features are shared with at least one other object
        non_min = []
        for i in true_idxs:
            if fcs[i] > 1:
                other_feats = [f for f in range(n_feat) if f != h and objs[i][f] == 1]
                # Check if at least one non-heard feature of i is shared by another object in the scene
                if any(sum(objs[j][f] for j in range(len(objs))) > 1 for f in other_feats):
                    non_min.append(i)
        if non_min and all(dups[i] == 1 for i in single_feat_sing):
            target_map[(objs_str, h)] = set(non_min)
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
    active_targets = [t for t in targets if t is not None]
    hits = [1.0 if c in t else 0.0 for c, t in zip(choices, active_targets)]
    return float(np.mean(hits))
