# name: complex_false_object_rate
# description: On word trials, the proportion of choices falling on false objects possessing high visual complexity (at least 2 features, the highest feature count among non-matching objects in the display).
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    lookup = {}
    for _, row in sub[["objects", "utterance"]].drop_duplicates().iterrows():
        objs = json.loads(row["objects"])
        utt = int(row["utterance"])
        false_indices = [i for i, o in enumerate(objs) if o[utt] == 0]
        if false_indices:
            false_fcs = [sum(objs[i]) for i in false_indices]
            max_f = max(false_fcs)
            if max_f >= 2:
                lookup[(row["objects"], row["utterance"])] = [1 if (objs[i][utt] == 0 and sum(objs[i]) == max_f) else 0 for i in range(len(objs))]
    obj_strs = sub["objects"].to_numpy()
    utts = sub["utterance"].to_numpy()
    choices = sub["choice"].to_numpy(dtype=int)
    picks = [lookup[(s, u)][c] for s, u, c in zip(obj_strs, utts, choices) if (s, u) in lookup and c < len(lookup[(s, u)])]
    return float(np.mean(picks)) if len(picks) > 0 else 0.0
