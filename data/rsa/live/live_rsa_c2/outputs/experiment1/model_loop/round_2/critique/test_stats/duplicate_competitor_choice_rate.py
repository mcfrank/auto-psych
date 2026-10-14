# name: duplicate_competitor_choice_rate
# description: Proportion of choices for the competitor class among utterance trials where the heard word is true of a unique singleton target and a competitor object that has duplicate copies in the scene.
def test_statistic(df):
    unique_pairs = df[['objects', 'utterance']].drop_duplicates()
    comp_map = {}
    for _, r in unique_pairs.iterrows():
        utt = r['utterance']
        if pd.isna(utt) or utt == -1:
            continue
        u = int(utt)
        objs = json.loads(r['objects'])
        classes = {}
        for i, o in enumerate(objs):
            if o[u] == 1:
                first = objs.index(o)
                classes[first] = classes.get(first, 0) + 1
        if len(classes) == 2:
            items = list(classes.items())
            c1, n1 = items[0]
            c2, n2 = items[1]
            cnt1, cnt2 = sum(objs[c1]), sum(objs[c2])
            if cnt1 < cnt2 and n1 == 1 and n2 > 1:
                comp_map[(r['objects'], u)] = c2
            elif cnt2 < cnt1 and n2 == 1 and n1 > 1:
                comp_map[(r['objects'], u)] = c1
    utts = df['utterance'].fillna(-1).astype(int)
    keys = list(zip(df['objects'], utts))
    targets = [comp_map.get(k, -1) for k in keys]
    mask = [t != -1 for t in targets]
    if not any(mask):
        return 0.0
    sub_df = df[mask]
    sub_targets = np.array([t for t in targets if t != -1])
    return float(np.mean(sub_df['choice'].to_numpy() == sub_targets))
