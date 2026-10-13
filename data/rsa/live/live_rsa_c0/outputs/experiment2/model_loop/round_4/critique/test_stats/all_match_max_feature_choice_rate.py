# name: all_match_max_feature_choice_rate
# description: Rate of choosing the object with the maximal feature count when the heard utterance is true of all objects in the display.
def test_statistic(df):
    unique_displays = df[["objects", "utterance"]].drop_duplicates()
    display_info = {}
    for r in unique_displays.itertuples(index=False):
        key = (r.objects, str(r.utterance))
        if pd.isna(r.utterance):
            display_info[key] = (False, set())
            continue
        objs = json.loads(r.objects)
        u = int(r.utterance)
        if all(obj[u] == 1 for obj in objs):
            fcounts = [sum(obj) for obj in objs]
            max_fc = max(fcounts)
            min_fc = min(fcounts)
            if max_fc != min_fc:
                max_indices = {i for i, fc in enumerate(fcounts) if fc == max_fc}
                display_info[key] = (True, max_indices)
                continue
        display_info[key] = (False, set())

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    choices = df["choice"].to_numpy()
    if not np.any(valid):
        return 0.0
    hits = [choices[i] in display_info[keys[i]][1] for i in range(len(keys)) if valid[i]]
    return float(np.mean(hits))
