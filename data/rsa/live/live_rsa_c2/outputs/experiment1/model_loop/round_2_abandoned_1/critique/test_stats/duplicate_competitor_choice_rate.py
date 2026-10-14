# name: duplicate_competitor_choice_rate
# description: Proportion of choices for the competitor class among utterance trials where the heard word is true of a unique singleton target and a competitor object that has duplicate copies in the scene.
def test_statistic(df):
    unique_pairs = df[['objects', 'utterance']].drop_duplicates()
    comp_map = {}
    for _, r in unique_pairs.iterrows():
        utt = r['utterance']
        if pd.isna(utt):
            continue
        u = int(utt)
        objs = json.loads(r['objects'])
        classes_with_u = {}
        for idx, o in enumerate(objs):
            if o[u] == 1:
                first = objs.index(o)
                classes_with_u[first] = classes_with_u.get(first, 0) + 1
        if len(classes_with_u) == 2:
            c_single = [c for c, count in classes_with_u.items() if count == 1]
            c_dup = [c for c, count in classes_with_u.items() if count > 1]
            if len(c_single) == 1 and len(c_dup) == 1:
                comp_map[(r['objects'], u)] = c_dup[0]
    utts = df['utterance'].fillna(-1).astype(int)
    keys = list(zip(df['objects'], utts))
    targets = [comp_map.get(k, -1) for k in keys]
    mask = [t != -1 for t in targets]
    if not any(mask):
        return 0.0
    sub_df = df[mask]
    sub_targets = np.array([t for t in targets if t != -1])
    return float(np.mean(sub_df['choice'].to_numpy() == sub_targets))
