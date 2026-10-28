# name: singleton_same_feature_count_vs_duplicated_true
# description: Rate of choosing the unique (singleton) word-true referent on heard-word trials where a singleton word-true referent competes against a duplicated pair of word-true referents with the exact same total number of features.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    target_map = {}
    for objs_str, heard in zip(scenes["objects"], scenes["heard"]):
        if pd.isna(heard):
            continue
        h = int(heard)
        objs = json.loads(objs_str)
        true_idxs = [i for i, o in enumerate(objs) if o[h] == 1]
        dups = [sum(1 for o2 in objs if o2 == objs[i]) for i in range(len(objs))]
        fcs = [sum(o) for o in objs]
        dup_true = [i for i in true_idxs if dups[i] > 1]
        sing_true = [i for i in true_idxs if dups[i] == 1]
        if dup_true and sing_true:
            dup_fcs = {fcs[i] for i in dup_true}
            same_fc_sing = [i for i in sing_true if fcs[i] in dup_fcs]
            if same_fc_sing:
                target_map[(objs_str, h)] = set(same_fc_sing)
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
