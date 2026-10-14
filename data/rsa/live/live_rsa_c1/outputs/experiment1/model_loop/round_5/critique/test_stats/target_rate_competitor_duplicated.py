# name: target_rate_competitor_duplicated
# description: Rate of choosing the minimal-feature target in ambiguous utterance trials where at least one competing matching object has an identical duplicate in the display.
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
            min_matches = [matches[k] for k, c in enumerate(counts) if c == min_c]
            if len(min_matches) == 1:
                target = min_matches[0]
                obj_tuples = [tuple(o) for o in objs]
                is_singleton = [obj_tuples.count(t) == 1 for t in obj_tuples]
                competitors = [m for m in matches if m != target]
                if any(not is_singleton[c] for c in competitors):
                    lookup[(r['objects'], str(r['utterance']))] = target
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    targets = np.array([lookup.get(k, -1) for k in keys])
    mask = targets != -1
    if not np.any(mask):
        return 0.0
    return float(np.mean(sub['choice'].to_numpy()[mask] == targets[mask]))
