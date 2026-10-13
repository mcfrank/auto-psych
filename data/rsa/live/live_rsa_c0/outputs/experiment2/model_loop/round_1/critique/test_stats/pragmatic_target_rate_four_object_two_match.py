# name: pragmatic_target_rate_four_object_two_match
# description: Rate of choosing the minimal-feature matching candidate (the pragmatic target) in four-object displays where the utterance matches exactly two candidates.
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
        matches = [i for i, obj in enumerate(objs) if obj[u] == 1]
        if len(matches) == 2:
            fc0 = sum(objs[matches[0]])
            fc1 = sum(objs[matches[1]])
            if fc0 != fc1:
                target_idx = matches[0] if fc0 < fc1 else matches[1]
                display_info[key] = (True, target_idx)
                continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
