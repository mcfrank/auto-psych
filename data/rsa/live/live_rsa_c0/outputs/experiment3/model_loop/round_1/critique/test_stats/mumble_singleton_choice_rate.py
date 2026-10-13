# name: mumble_singleton_choice_rate
# description: Rate of choosing the unique singleton object on mumble trials where duplicate objects are present in the display.
def test_statistic(df):
    unique_displays = df[["objects", "utterance"]].drop_duplicates()
    display_info = {}
    for r in unique_displays.itertuples(index=False):
        key = (r.objects, str(r.utterance))
        if not pd.isna(r.utterance):
            display_info[key] = (False, set())
            continue
        objs = json.loads(r.objects)
        obj_tuples = [tuple(o) for o in objs]
        counts = {t: obj_tuples.count(t) for t in set(obj_tuples)}
        if any(c > 1 for c in counts.values()) and any(c == 1 for c in counts.values()):
            singleton_indices = {i for i, t in enumerate(obj_tuples) if counts[t] == 1}
            display_info[key] = (True, singleton_indices)
            continue
        display_info[key] = (False, set())

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    choices = df["choice"].to_numpy()
    if not np.any(valid):
        return 0.0
    hits = [choices[i] in display_info[keys[i]][1] for i in range(len(keys)) if valid[i]]
    return float(np.mean(hits))
