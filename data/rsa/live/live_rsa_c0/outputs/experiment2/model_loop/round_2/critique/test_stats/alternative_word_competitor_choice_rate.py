# name: alternative_word_competitor_choice_rate
# description: Rate of choosing the competitor candidate possessing an unmentioned unique feature (an unambiguous alternative word) in displays with two matching candidates.
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
        matches = [i for i, obj in enumerate(objs) if obj[u] == 1]
        if len(matches) == 2:
            m0, m1 = matches[0], matches[1]
            n_feats = len(objs[0])
            col_sums = [sum(objs[i][f] for i in range(len(objs))) for f in range(n_feats)]
            m0_has_unique = any(objs[m0][f] == 1 and col_sums[f] == 1 for f in range(n_feats))
            m1_has_unique = any(objs[m1][f] == 1 and col_sums[f] == 1 for f in range(n_feats))
            if m0_has_unique != m1_has_unique:
                competitor_idx = m0 if m0_has_unique else m1
                display_info[key] = (True, competitor_idx)
                continue
        display_info[key] = (False, -1)

    keys = list(zip(df["objects"], df["utterance"].astype(str)))
    valid = np.array([display_info[k][0] for k in keys])
    competitors = np.array([display_info[k][1] for k in keys])
    if not np.any(valid):
        return 0.0
    return float(np.mean(df["choice"].to_numpy()[valid] == competitors[valid]))
