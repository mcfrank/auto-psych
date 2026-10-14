# name: target_rate_two_matching_objects
# description: Rate of choosing the minimal-feature target in utterance trials where the heard word matches exactly two objects with different feature counts.
def test_statistic(df):
    sub = df[df['query'] == 'utterance']
    uniques = sub[['objects', 'utterance']].drop_duplicates()
    lookup = {}
    for _, r in uniques.iterrows():
        objs = json.loads(r['objects'])
        utt = int(r['utterance'])
        matches = [i for i, o in enumerate(objs) if o[utt] == 1]
        if len(matches) == 2:
            counts = [sum(objs[i]) for i in matches]
            if counts[0] != counts[1]:
                target = matches[0] if counts[0] < counts[1] else matches[1]
                lookup[(r['objects'], str(r['utterance']))] = target
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    targets = np.array([lookup.get(k, -1) for k in keys])
    mask = targets != -1
    return float(np.mean(sub['choice'].to_numpy()[mask] == targets[mask])) if np.any(mask) else 0.0
