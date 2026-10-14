# name: competitor_choice_rate_large_feature_gap
# description: Rate of choosing the maximal-feature competitor in ambiguous utterance trials where the competitor has at least 3 more features than the minimal target.
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
            min_c, max_c = min(counts), max(counts)
            max_matches = [matches[k] for k, c in enumerate(counts) if c == max_c]
            if len(max_matches) == 1 and (max_c - min_c) >= 3:
                lookup[(r['objects'], str(r['utterance']))] = max_matches[0]
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    targets = np.array([lookup.get(k, -1) for k in keys])
    mask = targets != -1
    return float(np.mean(sub['choice'].to_numpy()[mask] == targets[mask])) if np.any(mask) else 0.0
