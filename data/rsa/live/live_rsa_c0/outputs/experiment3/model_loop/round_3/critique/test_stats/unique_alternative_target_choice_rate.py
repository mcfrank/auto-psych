# name: unique_alternative_target_choice_rate
# description: Rate of choosing the pragmatic target in displays where the target possesses a unique alternative feature while the competitor does not.
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
        if len(matches) == 2:
            m0, m1 = matches
            fc0 = sum(objs[m0])
            fc1 = sum(objs[m1])
            if fc0 != fc1:
                target = m0 if fc0 < fc1 else m1
                comp = m1 if fc0 < fc1 else m0
                n_feats = len(objs[0])
                feat_counts = [sum(o[k] for o in objs) for k in range(n_feats)]
                target_has_uniq = any(objs[target][k] == 1 and feat_counts[k] == 1 for k in range(n_feats))
                comp_has_uniq = any(objs[comp][k] == 1 and feat_counts[k] == 1 for k in range(n_feats))
                if target_has_uniq and (not comp_has_uniq):
                    display_info[key] = (True, target)
                    continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    targets = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == targets[valid]))
