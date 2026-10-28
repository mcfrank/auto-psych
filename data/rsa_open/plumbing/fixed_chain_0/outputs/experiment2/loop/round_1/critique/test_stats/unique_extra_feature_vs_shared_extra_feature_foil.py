# name: unique_extra_feature_vs_shared_extra_feature_foil
# description: Rate of choosing a word-true referent that possesses a scene-unique non-heard feature over a word-true referent with the same feature count whose non-heard feature is shared with a distractor, conditioned on heard-word trials where both types of referents compete.
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
        fcs = [sum(o) for o in objs]
        feat_totals = [sum(o[f] for o in objs) for f in range(n_feat)]
        has_unique_other = [
            any(objs[i][f] == 1 and feat_totals[f] == 1 for f in range(n_feat) if f != h)
            for i in range(len(objs))
        ]
        # Look for pairs of true objects with equal feature count where one has a unique extra feature and the other does not
        uniq_cands = [i for i in true_idxs if has_unique_other[i]]
        shared_cands = [i for i in true_idxs if not has_unique_other[i] and fcs[i] in {fcs[u] for u in uniq_cands}]
        if uniq_cands and shared_cands:
            matched_uniq = [i for i in uniq_cands if fcs[i] in {fcs[s] for s in shared_cands}]
            target_map[(objs_str, h)] = set(matched_uniq)
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
