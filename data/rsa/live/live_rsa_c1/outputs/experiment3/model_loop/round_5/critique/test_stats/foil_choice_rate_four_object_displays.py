# name: foil_choice_rate_four_object_displays
# description: Rate of choosing any non-matching foil object in utterance trials with four-object displays.
def test_statistic(df):
    sub = df[df['query'] == 'utterance']
    uniques = sub[['objects', 'utterance']].drop_duplicates()
    lookup = {}
    for _, r in uniques.iterrows():
        objs = json.loads(r['objects'])
        if len(objs) == 4:
            utt = int(r['utterance'])
            non_matches = {i for i, o in enumerate(objs) if o[utt] == 0}
            if non_matches:
                lookup[(r['objects'], str(r['utterance']))] = non_matches
    keys = list(zip(sub['objects'], sub['utterance'].astype(str)))
    choices = sub['choice'].to_numpy()
    hits = [int(c) in lookup[k] for c, k in zip(choices, keys) if k in lookup]
    return float(np.mean(hits)) if hits else 0.0
