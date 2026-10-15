# name: two_object_subset_target_rate
# description: On 2-object word trials where candidate objects have a strict subset-superset feature relationship, the proportion of choices for the minimal subset object (the pragmatic target).
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    lookup = {}
    for _, row in sub[["objects", "utterance"]].drop_duplicates().iterrows():
        objs = json.loads(row["objects"])
        if len(objs) != 2:
            continue
        utt = int(row["utterance"])
        o0, o1 = objs[0], objs[1]
        if o0[utt] == 1 and o1[utt] == 1:
            is_0 = all(o0[k] <= o1[k] for k in range(len(o0))) and sum(o0) < sum(o1)
            is_1 = all(o1[k] <= o0[k] for k in range(len(o0))) and sum(o1) < sum(o0)
            if is_0 or is_1:
                target = 0 if is_0 else 1
                lookup[(row["objects"], row["utterance"])] = target
    obj_strs = sub["objects"].to_numpy()
    utts = sub["utterance"].to_numpy()
    choices = sub["choice"].to_numpy(dtype=int)
    picks = [1 if c == lookup[(s, u)] else 0 for s, u, c in zip(obj_strs, utts, choices) if (s, u) in lookup]
    return float(np.mean(picks)) if len(picks) > 0 else 0.0
