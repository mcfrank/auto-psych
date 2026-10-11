# name: two_object_pragmatic_target_rate
# description: Rate of choosing the pragmatic target with fewer features on utterance trials with exactly two objects that both match the heard word.
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    mapping = {}
    for (obj_str, u) in sub[["objects", "utterance"]].drop_duplicates().itertuples(index=False):
        objs = json.loads(obj_str)
        if len(objs) == 2:
            u_int = int(u)
            if objs[0][u_int] == 1 and objs[1][u_int] == 1:
                c0, c1 = sum(objs[0]), sum(objs[1])
                if c0 != c1:
                    target = 0 if c0 < c1 else 1
                    mapping[(obj_str, u)] = target
    if not mapping:
        return 0.0
    matched = sub[[k in mapping for k in zip(sub["objects"], sub["utterance"])]]
    if len(matched) == 0:
        return 0.0
    is_target = [c == mapping[(o, u)] for c, o, u in zip(matched["choice"], matched["objects"], matched["utterance"])]
    return float(np.mean(is_target))
