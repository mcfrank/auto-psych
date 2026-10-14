# name: two_object_scalar_target_rate
# description: Proportion of choices for the minimal target object among two-object utterance trials where the heard word is true of both objects.
def test_statistic(df):
    unique_pairs = df[['objects', 'utterance']].drop_duplicates()
    target_map = {}
    for _, r in unique_pairs.iterrows():
        utt = r['utterance']
        if pd.isna(utt):
            continue
        u = int(utt)
        objs = json.loads(r['objects'])
        if len(objs) == 2 and objs[0][u] == 1 and objs[1][u] == 1:
            cnt0, cnt1 = sum(objs[0]), sum(objs[1])
            if cnt0 != cnt1:
                target_map[(r['objects'], u)] = 0 if cnt0 < cnt1 else 1
    utts = df['utterance'].fillna(-1).astype(int)
    keys = list(zip(df['objects'], utts))
    targets = [target_map.get(k, -1) for k in keys]
    mask = [t != -1 for t in targets]
    if not any(mask):
        return 0.0
    sub_df = df[mask]
    sub_targets = np.array([t for t in targets if t != -1])
    return float(np.mean(sub_df['choice'].to_numpy() == sub_targets))
