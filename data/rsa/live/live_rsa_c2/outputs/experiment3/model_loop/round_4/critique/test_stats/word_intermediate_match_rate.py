# name: word_intermediate_match_rate
# description: On word trials where matching candidates span at least three distinct feature counts, the proportion of choices falling on strictly intermediate-complexity matching objects.
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    lookup = {}
    for _, row in sub[["objects", "utterance"]].drop_duplicates().iterrows():
        objs = json.loads(row["objects"])
        utt = int(row["utterance"])
        matching = [i for i, o in enumerate(objs) if o[utt] == 1]
        matching_fcs = [sum(objs[i]) for i in matching]
        if len(matching_fcs) >= 3 and min(matching_fcs) < max(matching_fcs):
            min_m, max_m = min(matching_fcs), max(matching_fcs)
            flags = [1 if (objs[i][utt] == 1 and min_m < sum(objs[i]) < max_m) else 0 for i in range(len(objs))]
            if any(flags):
                lookup[(row["objects"], row["utterance"])] = flags
    obj_strs = sub["objects"].to_numpy()
    utts = sub["utterance"].to_numpy()
    choices = sub["choice"].to_numpy(dtype=int)
    picks = [lookup[(s, u)][c] for s, u, c in zip(obj_strs, utts, choices) if (s, u) in lookup]
    return float(np.mean(picks)) if len(picks) > 0 else 0.0
