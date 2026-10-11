# name: scalar_intermediate_competitor_rate
# description: Rate of choosing intermediate competitors on utterance trials where three or more objects match the heard word with a strict hierarchy of feature counts.
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    mapping = {}
    for (obj_str, u) in sub[["objects", "utterance"]].drop_duplicates().itertuples(index=False):
        objs = json.loads(obj_str)
        u_int = int(u)
        matches = [i for i, row in enumerate(objs) if row[u_int] == 1]
        if len(matches) >= 3:
            counts = [sum(objs[i]) for i in matches]
            u_counts = sorted(set(counts))
            if len(u_counts) >= 3:
                min_c, max_c = u_counts[0], u_counts[-1]
                inter_indices = {i for i in matches if min_c < sum(objs[i]) < max_c}
                mapping[(obj_str, u)] = inter_indices
    if not mapping:
        return 0.0
    matched = sub[[k in mapping for k in zip(sub["objects"], sub["utterance"])]]
    if len(matched) == 0:
        return 0.0
    is_inter = [c in mapping[(o, u)] for c, o, u in zip(matched["choice"], matched["objects"], matched["utterance"])]
    return float(np.mean(is_inter))
