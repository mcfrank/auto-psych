# name: mumble_max_feature_choice_rate
# description: Rate of choosing the object with the maximal number of features on mumble trials where objects differ in feature counts.
def test_statistic(df):
    unique_displays = df[["objects", "query"]].drop_duplicates()
    display_info = {}
    for r in unique_displays.itertuples(index=False):
        key = (r.objects, str(r.query))
        if r.query != "prior":
            display_info[key] = (False, -1)
            continue
        objs = json.loads(r.objects)
        objs_tuple = [tuple(o) for o in objs]
        fcs = [sum(o) for o in objs]
        max_fc = max(fcs)
        min_fc = min(fcs)
        if max_fc > min_fc:
            max_objs = [i for i, fc in enumerate(fcs) if fc == max_fc]
            first_idx = objs_tuple.index(objs_tuple[max_objs[0]])
            if all(objs_tuple[m] == objs_tuple[max_objs[0]] for m in max_objs):
                display_info[key] = (True, first_idx)
                continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["query"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
