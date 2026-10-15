# name: non_nested_fewer_features_rate
# description: On word trials where matching candidates have different feature counts and cross-cutting (non-nested) features, the proportion of choices for the candidate with the fewest total features.
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    lookup = {}
    for _, row in sub[["objects", "utterance"]].drop_duplicates().iterrows():
        objs = json.loads(row["objects"])
        utt = int(row["utterance"])
        matching = [i for i, o in enumerate(objs) if o[utt] == 1]
        fcs = [sum(objs[i]) for i in matching]
        if len(matching) >= 2 and min(fcs) < max(fcs):
            min_i = matching[int(np.argmin(fcs))]
            max_i = matching[int(np.argmax(fcs))]
            o_min, o_max = objs[min_i], objs[max_i]
            if not all(o_min[k] <= o_max[k] for k in range(len(o_min))):
                min_fc = min(fcs)
                lookup[(row["objects"], row["utterance"])] = [1 if (objs[i][utt] == 1 and sum(objs[i]) == min_fc) else 0 for i in range(len(objs))]
    obj_strs = sub["objects"].to_numpy()
    utts = sub["utterance"].to_numpy()
    choices = sub["choice"].to_numpy(dtype=int)
    picks = [lookup[(s, u)][c] for s, u, c in zip(obj_strs, utts, choices) if (s, u) in lookup and c < len(lookup[(s, u)])]
    return float(np.mean(picks)) if len(picks) > 0 else 0.0
