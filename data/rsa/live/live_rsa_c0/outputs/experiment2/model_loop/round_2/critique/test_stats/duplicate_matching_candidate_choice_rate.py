# name: duplicate_matching_candidate_choice_rate
# description: Rate of choosing a duplicated candidate in displays where the heard utterance matches both a duplicated object and a unique object.
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
        matches = [i for i, obj in enumerate(objs) if obj[u] == 1]
        match_tuples = [tuple(objs[i]) for i in matches]
        counts = {t: match_tuples.count(t) for t in set(match_tuples)}
        if any(c > 1 for c in counts.values()) and any(c == 1 for c in counts.values()):
            dup_indices = {matches[i] for i, t in enumerate(match_tuples) if counts[t] > 1}
            display_info[key] = (True, dup_indices)
            continue
        display_info[key] = (False, set())

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    choices = df["choice"].to_numpy()
    if not np.any(valid):
        return 0.0
    hits = [choices[i] in display_info[keys[i]][1] for i in range(len(keys)) if valid[i]]
    return float(np.mean(hits))
