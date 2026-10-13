# name: three_match_intermediate_feature_rate
# description: Rate of choosing the intermediate-feature candidate in displays where exactly three candidates match the heard utterance with strictly distinct feature counts.
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
        if len(matches) == 3:
            fcs = [sum(objs[i]) for i in matches]
            if len(set(fcs)) == 3:
                sorted_matches = [m for _, m in sorted(zip(fcs, matches))]
                mid_idx = sorted_matches[1]
                display_info[key] = (True, mid_idx)
                continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
