# name: standard_scalar_logical_choice_rate
# description: Rate of choosing the logical competitor on three-object scalar implicature displays with one foil, one target, and one competitor matching the heard word.
def test_statistic(df):
    sub = df[df["query"] == "utterance"]
    mapping = {}
    for (obj_str, u) in sub[["objects", "utterance"]].drop_duplicates().itertuples(index=False):
        objs = json.loads(obj_str)
        if len(objs) == 3:
            u_int = int(u)
            matches = [i for i, row in enumerate(objs) if row[u_int] == 1]
            if len(matches) == 2:
                c0, c1 = sum(objs[matches[0]]), sum(objs[matches[1]])
                if c0 != c1:
                    foil = [i for i in range(3) if i not in matches][0]
                    if objs[foil][u_int] == 0:
                        logical_idx = matches[0] if c0 > c1 else matches[1]
                        mapping[(obj_str, u)] = logical_idx
    if not mapping:
        return 0.0
    matched = sub[[k in mapping for k in zip(sub["objects"], sub["utterance"])]]
    if len(matched) == 0:
        return 0.0
    is_logical = [c == mapping[(o, u)] for c, o, u in zip(matched["choice"], matched["objects"], matched["utterance"])]
    return float(np.mean(is_logical))
