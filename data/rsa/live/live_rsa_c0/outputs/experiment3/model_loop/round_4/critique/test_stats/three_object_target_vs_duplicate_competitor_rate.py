# name: three_object_target_vs_duplicate_competitor_rate
# description: Rate of choosing the pragmatic target in three-object displays where the heard utterance matches the target and a duplicated competitor.
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
        objs_tuple = [tuple(o) for o in objs]
        matches = [i for i, o in enumerate(objs) if o[u] == 1]
        if len(matches) >= 2:
            fcs = [sum(objs[i]) for i in matches]
            min_fc = min(fcs)
            targets = [i for i in matches if sum(objs[i]) == min_fc]
            comps = [i for i in matches if sum(objs[i]) > min_fc]
            if len(targets) == 1 and len(comps) >= 2:
                comp_tuples = [objs_tuple[i] for i in comps]
                if len(set(comp_tuples)) == 1:
                    display_info[key] = (True, targets[0])
                    continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
