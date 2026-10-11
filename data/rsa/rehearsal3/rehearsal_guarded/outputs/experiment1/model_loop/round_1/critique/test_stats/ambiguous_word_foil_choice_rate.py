# name: ambiguous_word_foil_choice_rate
# description: Rate of choosing non-matching foil objects on utterance trials where the heard word is true of two or more objects.
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    mapping = {}
    for (obj_str, u) in sub[["objects", "utterance"]].drop_duplicates().itertuples(index=False):
        objs = json.loads(obj_str)
        u_int = int(u)
        matches = [i for i, row in enumerate(objs) if row[u_int] == 1]
        if len(matches) >= 2 and len(matches) < len(objs):
            foils = {i for i, row in enumerate(objs) if row[u_int] == 0}
            mapping[(obj_str, u)] = foils
    if not mapping:
        return 0.0
    matched = sub[[k in mapping for k in zip(sub["objects"], sub["utterance"])]]
    if len(matched) == 0:
        return 0.0
    is_foil = [c in mapping[(o, u)] for c, o, u in zip(matched["choice"], matched["objects"], matched["utterance"])]
    return float(np.mean(is_foil))
