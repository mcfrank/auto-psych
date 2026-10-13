# name: pragmatic_target_rate_two_object
# description: Rate of choosing the minimal-feature matching object (the pragmatic target) in two-object displays where an utterance is referentially ambiguous.
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
            if fc0 != fc1:
                target_idx = 0 if fc0 < fc1 else 1
                display_info[key] = (True, target_idx)
                continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
