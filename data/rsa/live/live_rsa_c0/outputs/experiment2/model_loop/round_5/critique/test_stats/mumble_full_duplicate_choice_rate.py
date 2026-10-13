# name: mumble_full_duplicate_choice_rate
# description: Rate of choosing a duplicated candidate possessing all features on mumble trials where identical objects are present.
def test_statistic(df):
    unique_displays = df[["objects", "utterance"]].drop_duplicates()
    display_info = {}
    for r in unique_displays.itertuples(index=False):
        key = (r.objects, str(r.utterance))
        if not pd.isna(r.utterance):
            display_info[key] = (False, set())
            continue
        objs = json.loads(r.objects)
        n_feats = len(objs[0])
        obj_tuples = [tuple(o) for o in objs]
        counts = {t: obj_tuples.count(t) for t in set(obj_tuples)}
        full_tuple = tuple([1] * n_feats)
        if full_tuple in counts and counts[full_tuple] > 1:
            full_indices = {i for i, t in enumerate(obj_tuples) if t == full_tuple}
            display_info[key] = (True, full_indices)
            continue
        display_info[key] = (False, set())

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    choices = df["choice"].to_numpy()
    if not np.any(valid):
        return 0.0
    hits = [choices[i] in display_info[keys[i]][1] for i in range(len(keys)) if valid[i]]
    return float(np.mean(hits))
