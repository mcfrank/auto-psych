# name: target_rate_redundant_feature_utterance
# description: Rate of choosing the minimal-feature target in ambiguous utterance trials where the heard word is a redundant feature sharing its exact object extension with another feature.
def test_statistic(df):
    sub = df[df['query'] == 'utterance']
    uniques = sub[['objects', 'utterance']].drop_duplicates()
    lookup = {}
    for _, r in uniques.iterrows():
        objs = np.array(json.loads(r['objects']))
        utt = int(r['utterance'])
        matches = np.where(objs[:, utt] == 1)[0]
        if len(matches) >= 2:
            is_red = any(j != utt and np.array_equal(objs[:, j], objs[:, utt]) for j in range(objs.shape[1]))
            if is_red:
                counts = [np.sum(objs[i]) for i in matches]
                min_c = min(counts)
                min_matches = [matches[k] for k, c in enumerate(counts) if c == min_c]
                if len(min_matches) == 1:
                    lookup[(r['objects'], str(r['utterance']))] = min_matches[0]
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    targets = np.array([lookup.get(k, -1) for k in keys])
    mask = targets != -1
    return float(np.mean(sub['choice'].to_numpy()[mask] == targets[mask])) if np.any(mask) else 0.0
