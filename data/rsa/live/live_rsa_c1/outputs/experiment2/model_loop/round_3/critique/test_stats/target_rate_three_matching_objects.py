# name: target_rate_three_matching_objects
# description: Rate of choosing the minimal-feature target in utterance trials where the heard word matches exactly three objects with differing feature counts.
def test_statistic(df):
    sub = df[df['query'] == 'utterance']
    uniques = sub[['objects', 'utterance']].drop_duplicates()
    lookup = {}
    for _, r in uniques.iterrows():
        objs = json.loads(r['objects'])
        utt = int(r['utterance'])
        matches = [i for i, o in enumerate(objs) if o[utt] == 1]
        if len(matches) == 3:
            counts = [sum(objs[i]) for i in matches]
            min_c, max_c = min(counts), max(counts)
            if min_c < max_c:
                min_matches = [matches[k] for k, c in enumerate(counts) if c == min_c]
                if len(min_matches) == 1:
                    lookup[(r['objects'], str(r['utterance']))] = min_matches[0]
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    targets = np.array([lookup.get(k, -1) for k in keys])
    mask = targets != -1
    return float(np.mean(sub['choice'].to_numpy()[mask] == targets[mask])) if np.any(mask) else 0.0
