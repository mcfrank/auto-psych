# name: prior_duplicate_background_points
# description: On prior trials containing exactly one singleton among identical duplicate distractors, the total points allocated across all duplicate distractor objects combined.
def test_statistic(df):
    table = {}
    distinct = df[["objects", "heard"]].drop_duplicates()
    for _, sc in distinct.iterrows():
        if not pd.isna(sc["heard"]):
            continue
        objs = json.loads(sc["objects"])
        counts = {tuple(o): objs.count(o) for o in objs}
        singletons = [i for i, o in enumerate(objs) if counts[tuple(o)] == 1]
        duplicates = [i for i, o in enumerate(objs) if counts[tuple(o)] > 1]
        if len(singletons) == 1 and duplicates:
            table[sc["objects"]] = duplicates
    pts_list = [json.loads(p) if isinstance(p, str) else p for p in df["points"]]
    total, count = 0.0, 0
    for idx, (h, p) in enumerate(zip(df["heard"], pts_list)):
        if not pd.isna(h):
            continue
        key = df["objects"].iloc[idx]
        if key in table:
            total += sum(p[i] for i in table[key])
            count += 1
    return float(total / max(count, 1))
