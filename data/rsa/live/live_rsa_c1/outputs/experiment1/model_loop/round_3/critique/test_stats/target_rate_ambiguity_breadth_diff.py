# name: target_rate_ambiguity_breadth_diff
# description: Difference in minimal-feature target choice rate between narrow ambiguity (word fits exactly 2 objects) and broad ambiguity (word fits 3 or more objects).
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
                lookup[(r['objects'], str(r['utterance']))] = (len(matches), min_matches[0])
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    choices = sub['choice'].to_numpy()
    hits_2 = []
    hits_3p = []
    for c, k in zip(choices, keys):
        if k in lookup:
            n_m, target = lookup[k]
            if n_m == 2:
                hits_2.append(c == target)
            else:
                hits_3p.append(c == target)
    rate_2 = np.mean(hits_2) if hits_2 else 0.0
    rate_3p = np.mean(hits_3p) if hits_3p else 0.0
    return float(rate_2 - rate_3p)
