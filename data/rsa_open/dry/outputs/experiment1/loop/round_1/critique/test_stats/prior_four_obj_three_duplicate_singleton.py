# name: prior_four_obj_three_duplicate_singleton
# description: On 4-object prior trials containing a unique singleton alongside three identical duplicate distractors, the mean points allocated to the singleton object.
def test_statistic(df):
    table = {}
    distinct = df[["objects", "heard"]].drop_duplicates()
    for _, sc in distinct.iterrows():
        if not pd.isna(sc["heard"]):
            continue
        objs = json.loads(sc["objects"])
        if len(objs) == 4:
            counts = {tuple(o): objs.count(o) for o in objs}
            if sorted(counts.values()) == [1, 3]:
                s_idx = [i for i, o in enumerate(objs) if counts[tuple(o)] == 1][0]
                table[sc["objects"]] = s_idx
    pts_list = [json.loads(p) if isinstance(p, str) else p for p in df["points"]]
    total, count = 0.0, 0
    for idx, (h, p) in enumerate(zip(df["heard"], pts_list)):
        if not pd.isna(h):
            continue
        key = df["objects"].iloc[idx]
        if key in table:
            total += p[table[key]]
            count += 1
    return float(total / max(count, 1))
