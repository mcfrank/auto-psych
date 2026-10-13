# name: three_object_distractor_intrusion_rate
# description: Rate of choosing the non-matching distractor (solitary foil) in three-object displays where two objects match the utterance.
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
        matches = [i for i, obj in enumerate(objs) if obj[u] == 1]
        if len(matches) == 2:
            foil_idx = [i for i in range(3) if i not in matches][0]
            display_info[key] = (True, foil_idx)
            continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    foils = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == foils[valid]))
