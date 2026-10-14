# name: non_redundant_competitor_choice_rate
# description: Proportion of choices for the overspecified competitor object in utterance trials where all alternative features of the competitor are strictly non-redundant across the display.
def test_statistic(df):
    unique_pairs = df[['objects', 'utterance']].drop_duplicates()
    comp_map = {}
    for _, r in unique_pairs.iterrows():
        utt = r['utterance']
        if pd.isna(utt) or utt == -1:
            continue
        u = int(utt)
        objs = json.loads(r['objects'])
        n_feats = len(objs[0])
        cols = [[objs[i][f] for i in range(len(objs))] for f in range(n_feats)]
        classes = {}
        for i, o in enumerate(objs):
            if o[u] == 1:
                first = objs.index(o)
                classes[first] = classes.get(first, 0) + 1
        if len(classes) == 2:
            c1, c2 = list(classes.keys())
            cnt1, cnt2 = sum(objs[c1]), sum(objs[c2])
            if cnt1 != cnt2:
                comp = c1 if cnt1 > cnt2 else c2
                comp_other_feats = [f for f in range(n_feats) if f != u and objs[comp][f] == 1]
                has_red = False
                for f1 in comp_other_feats:
                    for f2 in comp_other_feats:
                        if f1 < f2 and cols[f1] == cols[f2]:
                            has_red = True
                            break
                if not has_red and len(comp_other_feats) >= 1:
                    comp_map[(r['objects'], u)] = comp
    utts = df['utterance'].fillna(-1).astype(int)
    keys = list(zip(df['objects'], utts))
    targets = [comp_map.get(k, -1) for k in keys]
    mask = [t != -1 for t in targets]
    if not any(mask):
        return 0.0
    sub_df = df[mask]
    sub_targets = np.array([t for t in targets if t != -1])
    return float(np.mean(sub_df['choice'].to_numpy() == sub_targets))
