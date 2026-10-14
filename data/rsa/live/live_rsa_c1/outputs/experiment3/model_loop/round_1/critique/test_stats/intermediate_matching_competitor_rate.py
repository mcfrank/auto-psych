# name: intermediate_matching_competitor_rate
# description: Rate of choosing an intermediate-feature competitor in utterance trials where the heard word matches three or more objects with differing feature counts.
def test_statistic(df):
    sub = df[df['query'] == 'utterance']
    uniques = sub[['objects', 'utterance']].drop_duplicates()
    lookup = {}
    for _, r in uniques.iterrows():
        objs = json.loads(r['objects'])
        utt = int(r['utterance'])
        matches = [i for i, o in enumerate(objs) if o[utt] == 1]
        if len(matches) >= 3:
            counts = [sum(objs[i]) for i in matches]
            min_c, max_c = min(counts), max(counts)
            if min_c < max_c:
                inter = {matches[k] for k, c in enumerate(counts) if min_c < c < max_c}
                if inter:
                    lookup[(r['objects'], str(r['utterance']))] = inter
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    choices = sub['choice'].to_numpy()
    hits = [int(c) in lookup[k] for c, k in zip(choices, keys) if k in lookup]
    return float(np.mean(hits)) if hits else 0.0
