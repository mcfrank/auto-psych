# name: false_feature_choice_rate
# description: Proportion of choices for an object of which the heard utterance is false across all utterance trials.
def test_statistic(df):
    unique_pairs = df[['objects', 'utterance']].drop_duplicates()
    false_map = {}
    for _, r in unique_pairs.iterrows():
        utt = r['utterance']
        if pd.isna(utt):
            continue
        u = int(utt)
        objs = json.loads(r['objects'])
        false_classes = set()
        for idx, o in enumerate(objs):
            if o[u] == 0:
                false_classes.add(objs.index(o))
        false_map[(r['objects'], u)] = false_classes
    utts = df['utterance'].fillna(-1).astype(int)
    keys = list(zip(df['objects'], utts))
    choices = df['choice'].to_numpy()
    is_false = [k in false_map and c in false_map[k] for k, c in zip(keys, choices)]
    utt_mask = [k in false_map for k in keys]
    if not any(utt_mask):
        return 0.0
    return float(np.sum(is_false) / np.sum(utt_mask))
