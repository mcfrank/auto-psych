# name: two_object_target_vs_full_competitor_rate
# description: Rate of choosing the minimal-feature pragmatic target in two-object displays where the utterance matches both objects and the competitor possesses all features.
def test_statistic(df):
    unique_displays = df[["objects", "utterance"]].drop_duplicates()
    display_info = {}
    for r in unique_displays.itertuples(index=False):
        key = (r.objects, str(r.utterance))
        if pd.isna(r.utterance):
            display_info[key] = (False, -1)
            continue
        objs = json.loads(r.objects)
        if len(objs) != 2:
            display_info[key] = (False, -1)
            continue
        u = int(r.utterance)
        if objs[0][u] == 1 and objs[1][u] == 1:
            n_feats = len(objs[0])
            fc0 = sum(objs[0])
            fc1 = sum(objs[1])
            is_full0 = fc0 == n_feats
            is_full1 = fc1 == n_feats
            if is_full0 != is_full1:
                target_idx = 1 if is_full0 else 0
                display_info[key] = (True, target_idx)
                continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
