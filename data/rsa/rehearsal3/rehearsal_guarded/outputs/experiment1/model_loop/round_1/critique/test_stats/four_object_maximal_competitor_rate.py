# name: four_object_maximal_competitor_rate
# description: Rate of choosing the maximal four-feature competitor on utterance trials in four-object displays where the heard word matches both a one-feature target and a four-feature competitor.
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    mapping = {}
    for (obj_str, u) in sub[["objects", "utterance"]].drop_duplicates().itertuples(index=False):
        objs = json.loads(obj_str)
        if len(objs) == 4:
            u_int = int(u)
            matches = [i for i, row in enumerate(objs) if row[u_int] == 1]
            if len(matches) >= 2:
                match_counts = {i: sum(objs[i]) for i in matches}
                if 1 in match_counts.values() and 4 in match_counts.values():
                    max_idx = [i for i, c in match_counts.items() if c == 4][0]
                    mapping[(obj_str, u)] = max_idx
    if not mapping:
        return 0.0
    matched = sub[[k in mapping for k in zip(sub["objects"], sub["utterance"])]]
    if len(matched) == 0:
        return 0.0
    is_max = [c == mapping[(o, u)] for c, o, u in zip(matched["choice"], matched["objects"], matched["utterance"])]
    return float(np.mean(is_max))
