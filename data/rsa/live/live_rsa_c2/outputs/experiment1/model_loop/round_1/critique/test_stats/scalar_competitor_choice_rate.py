# name: scalar_competitor_choice_rate
# description: Proportion of trials choosing the overspecified competitor object among utterance trials where the heard word is true of both a minimal target and a competitor with strictly more features.
def test_statistic(df):
    unique_pairs = df[['objects', 'utterance']].drop_duplicates()
    comp_map = {}
    for _, r in unique_pairs.iterrows():
        utt = r['utterance']
        if pd.isna(utt):
            continue
        u = int(utt)
        objs = json.loads(r['objects'])
        classes = {}
        for i, o in enumerate(objs):
            if o[u] == 1:
                first = objs.index(o)
                classes[first] = sum(o)
        if len(classes) == 2:
            c1, c2 = list(classes.keys())
            cnt1, cnt2 = classes[c1], classes[c2]
            if cnt1 != cnt2:
                comp_map[(r['objects'], u)] = c1 if cnt1 > cnt2 else c2
    utts = df['utterance'].fillna(-1).astype(int)
    keys = list(zip(df['objects'], utts))
    targets = [comp_map.get(k, -1) for k in keys]
    mask = [t != -1 for t in targets]
    if not any(mask):
        return 0.0
    sub_df = df[mask]
    sub_targets = np.array([t for t in targets if t != -1])
    return float(np.mean(sub_df['choice'].to_numpy() == sub_targets))
