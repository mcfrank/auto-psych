# name: unambiguous_word_accuracy
# description: Rate of choosing the uniquely matching referent on utterance trials where the heard word is true of exactly one object.
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    mapping = {}
    for (obj_str, u) in sub[["objects", "utterance"]].drop_duplicates().itertuples(index=False):
        objs = json.loads(obj_str)
        u_int = int(u)
        matches = [i for i, row in enumerate(objs) if row[u_int] == 1]
        if len(matches) == 1:
            mapping[(obj_str, u)] = matches[0]
    if not mapping:
        return 0.0
    matched = sub[[k in mapping for k in zip(sub["objects"], sub["utterance"])]]
    if len(matched) == 0:
        return 0.0
    correct = [c == mapping[(o, u)] for c, o, u in zip(matched["choice"], matched["objects"], matched["utterance"])]
    return float(np.mean(correct))
