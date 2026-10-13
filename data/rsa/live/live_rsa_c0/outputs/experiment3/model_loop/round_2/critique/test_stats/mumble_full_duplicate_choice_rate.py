# name: mumble_full_duplicate_choice_rate
# description: Rate of choosing a duplicated candidate possessing all features on mumble trials where identical objects are present.
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
        n_feats = len(objs[0])
        full_dups = [i for i, o in enumerate(objs) if sum(o) == n_feats and objs_tuple.count(objs_tuple[i]) > 1]
        if full_dups:
            first_idx = objs_tuple.index(objs_tuple[full_dups[0]])
            display_info[key] = (True, first_idx)
            continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["query"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
