# name: three_object_one_feature_target_vs_two_feature_competitor_rate
# description: Rate of choosing the single-feature pragmatic target in three-object displays where the heard utterance matches exactly two candidates and the competitor possesses two features.
def test_statistic(df):
    unique_displays = df[["objects", "utterance"]].drop_duplicates()
    display_info = {}
    for r in unique_displays.itertuples(index=False):
        key = (r.objects, str(r.utterance))
        if pd.isna(r.utterance):
            display_info[key] = (False, -1)
            continue
        objs = json.loads(r.objects)
        if len(objs) != 3:
            display_info[key] = (False, -1)
            continue
        u = int(r.utterance)
        matches = [i for i, o in enumerate(objs) if o[u] == 1]
        if len(matches) == 2:
            m0, m1 = matches
            fc0 = sum(objs[m0])
            fc1 = sum(objs[m1])
            if (fc0 == 1 and fc1 == 2) or (fc0 == 2 and fc1 == 1):
                target_idx = m0 if fc0 == 1 else m1
                display_info[key] = (True, target_idx)
                continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
