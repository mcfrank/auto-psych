# name: target_rate_scaling_4obj_vs_2obj
# description: Difference in minimal-feature target choice rate between 4-object displays and 2-object displays in utterance trials where the heard word matches two objects.
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
                tgt = matches[0] if counts[0] < counts[1] else matches[1]
                lookup[(r['objects'], str(r['utterance']))] = (len(objs), tgt)
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    choices = sub['choice'].to_numpy()
    hits_4 = []
    hits_2 = []
    for c, k in zip(choices, keys):
        if k in lookup:
            n_objs, target = lookup[k]
            if n_objs == 4:
                hits_4.append(c == target)
            elif n_objs == 2:
                hits_2.append(c == target)
    rate_4 = np.mean(hits_4) if hits_4 else 0.0
    rate_2 = np.mean(hits_2) if hits_2 else 0.0
    return float(rate_4 - rate_2)
