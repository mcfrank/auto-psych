# name: target_rate_large_feature_gap
# description: Rate of choosing the minimal-feature target in ambiguous utterance trials where the competing matching object has at least 3 more features than the target.
def test_statistic(df):
    sub = df[df['query'] == 'utterance']
    uniques = sub[['objects', 'utterance']].drop_duplicates()
    lookup = {}
    for _, r in uniques.iterrows():
        objs = json.loads(r['objects'])
        utt = int(r['utterance'])
        matches = [i for i, o in enumerate(objs) if o[utt] == 1]
        if len(matches) >= 2:
            counts = [sum(objs[i]) for i in matches]
            min_c = min(counts)
            max_c = max(counts)
            min_matches = [matches[k] for k, c in enumerate(counts) if c == min_c]
            if len(min_matches) == 1 and (max_c - min_c) >= 3:
                lookup[(r['objects'], str(r['utterance']))] = min_matches[0]
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    targets = np.array([lookup.get(k, -1) for k in keys])
    mask = targets != -1
    if not np.any(mask):
        return 0.0
    return float(np.mean(sub['choice'].to_numpy()[mask] == targets[mask]))
