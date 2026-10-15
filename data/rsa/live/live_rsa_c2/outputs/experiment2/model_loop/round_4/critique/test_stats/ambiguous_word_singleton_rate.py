# name: ambiguous_word_singleton_rate
# description: On word trials where the heard utterance is true of all objects in the display and candidates include both a unique singleton and duplicate objects, the proportion of choices for the singleton object.
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    lookup = {}
    for _, row in sub[["objects", "utterance"]].drop_duplicates().iterrows():
        objs = json.loads(row["objects"])
        utt = int(row["utterance"])
        if all(o[utt] == 1 for o in objs):
            tuples = [tuple(o) for o in objs]
            counts = [tuples.count(o) for o in tuples]
            if 1 in counts and any(c > 1 for c in counts):
                lookup[(row["objects"], row["utterance"])] = [1 if counts[i] == 1 else 0 for i in range(len(objs))]
    obj_strs = sub["objects"].to_numpy()
    utts = sub["utterance"].to_numpy()
    choices = sub["choice"].to_numpy(dtype=int)
    picks = [lookup[(s, u)][c] for s, u, c in zip(obj_strs, utts, choices) if (s, u) in lookup and c < len(lookup[(s, u)])]
    return float(np.mean(picks)) if len(picks) > 0 else 0.0
