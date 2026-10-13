# name: two_object_one_feature_target_vs_three_feature_competitor_rate
# description: Rate of choosing the single-feature pragmatic target in two-object displays where the heard utterance matches both objects and the competitor possesses three features.
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
            fc0 = sum(objs[0])
            fc1 = sum(objs[1])
            if (fc0 == 1 and fc1 == 3) or (fc0 == 3 and fc1 == 1):
                t_idx = 0 if fc0 == 1 else 1
                display_info[key] = (True, t_idx)
                continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
