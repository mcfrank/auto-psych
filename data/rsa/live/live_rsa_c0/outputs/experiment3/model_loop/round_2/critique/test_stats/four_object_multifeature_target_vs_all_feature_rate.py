# name: four_object_multifeature_target_vs_all_feature_rate
# description: Rate of choosing the multi-feature pragmatic target possessing at least two features in four-object displays where the utterance matches exactly two candidates and the competitor possesses all features.
def test_statistic(df):
    unique_displays = df[["objects", "utterance"]].drop_duplicates()
    display_info = {}
    for r in unique_displays.itertuples(index=False):
        key = (r.objects, str(r.utterance))
        if pd.isna(r.utterance):
            display_info[key] = (False, -1)
            continue
        objs = json.loads(r.objects)
        if len(objs) != 4:
            display_info[key] = (False, -1)
            continue
        u = int(r.utterance)
        matches = [i for i, o in enumerate(objs) if o[u] == 1]
        if len(matches) == 2:
            m0, m1 = matches
            n_feats = len(objs[0])
            fc0 = sum(objs[m0])
            fc1 = sum(objs[m1])
            is_full0 = (fc0 == n_feats)
            is_full1 = (fc1 == n_feats)
            if (is_full0 or is_full1) and min(fc0, fc1) >= 2:
                target_idx = m1 if is_full0 else m0
                display_info[key] = (True, target_idx)
                continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
