# name: uninformative_word_min_feature_rate
# description: Rate of choosing the minimal-feature object in utterance trials where the heard word is true of every object in the display.
def test_statistic(df):
    sub = df[df['query'] == 'utterance']
    uniques = sub[['objects', 'utterance']].drop_duplicates()
    lookup = {}
    for _, r in uniques.iterrows():
        objs = json.loads(r['objects'])
        utt = int(r['utterance'])
        matches = [i for i, o in enumerate(objs) if o[utt] == 1]
        if len(matches) == len(objs):
            counts = [sum(o) for o in objs]
            min_c = min(counts)
            min_matches = [i for i, c in enumerate(counts) if c == min_c]
            if len(min_matches) == 1:
                lookup[(r['objects'], str(r['utterance']))] = min_matches[0]
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    targets = np.array([lookup.get(k, -1) for k in keys])
    mask = targets != -1
    if not np.any(mask):
        return 0.0
    return float(np.mean(sub['choice'].to_numpy()[mask] == targets[mask]))
