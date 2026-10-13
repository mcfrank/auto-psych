# name: all_match_max_feature_choice_rate
# description: Rate of choosing the object with the maximal number of features in displays with at least three objects where the heard utterance matches all objects.
def test_statistic(df):
    unique_displays = df[["objects", "utterance"]].drop_duplicates()
    display_info = {}
    for r in unique_displays.itertuples(index=False):
        key = (r.objects, str(r.utterance))
        if pd.isna(r.utterance):
            display_info[key] = (False, -1)
            continue
        objs = json.loads(r.objects)
        u = int(r.utterance)
        matches = [i for i, o in enumerate(objs) if o[u] == 1]
        if len(matches) == len(objs) and len(objs) >= 3:
            fcs = [sum(o) for o in objs]
            max_fc = max(fcs)
            min_fc = min(fcs)
            if max_fc > min_fc:
                max_indices = [i for i, fc in enumerate(fcs) if fc == max_fc]
                objs_tuple = [tuple(o) for o in objs]
                first_idx = objs_tuple.index(objs_tuple[max_indices[0]])
                if all(objs_tuple[i] == objs_tuple[max_indices[0]] for i in max_indices):
                    display_info[key] = (True, first_idx)
                    continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
