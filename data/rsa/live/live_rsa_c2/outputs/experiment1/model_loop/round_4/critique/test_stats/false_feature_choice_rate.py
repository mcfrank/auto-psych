# name: false_feature_choice_rate
# description: Proportion of choices for an object of which the heard utterance is false (literal semantic violation) across all utterance trials.
def test_statistic(df):
    unique_pairs = df[['objects', 'utterance']].drop_duplicates()
    false_map = {}
    for _, r in unique_pairs.iterrows():
        utt = r['utterance']
        if pd.isna(utt) or utt == -1:
            continue
        u = int(utt)
        objs = json.loads(r['objects'])
        false_indices = {objs.index(o) for o in objs if o[u] == 0}
        false_map[(r['objects'], u)] = false_indices
    utts = df['utterance'].fillna(-1).astype(int)
    keys = list(zip(df['objects'], utts))
    targets = [false_map.get(k, set()) for k in keys]
    mask = [len(s) > 0 for s in targets]
    if not any(mask):
        return 0.0
    sub_df = df[mask]
    sub_targets = [targets[i] for i, m in enumerate(mask) if m]
    chosen = [c in s for c, s in zip(sub_df['choice'], sub_targets)]
    return float(np.mean(chosen))
