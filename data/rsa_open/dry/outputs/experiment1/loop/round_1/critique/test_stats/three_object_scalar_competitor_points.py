# name: three_object_scalar_competitor_points
# description: On 3-object communicative trials with a unique minimal target and an uninformative competitor satisfying the message, the mean points allocated to the competitor object.
def test_statistic(df):
    table = {}
    distinct = df[["objects", "heard"]].drop_duplicates()
    for _, sc in distinct.iterrows():
        if pd.isna(sc["heard"]):
            continue
        h = int(sc["heard"])
        objs = json.loads(sc["objects"])
        if len(objs) == 3:
            sat = [i for i, o in enumerate(objs) if o[h] == 1]
            if len(sat) > 1:
                fc = [sum(objs[i]) for i in sat]
                min_c = min(fc)
                if fc.count(min_c) == 1:
                    target_idx = sat[fc.index(min_c)]
                    comp_indices = [i for i in sat if i != target_idx]
                    table[(sc["objects"], h)] = comp_indices
    pts_list = [json.loads(p) if isinstance(p, str) else p for p in df["points"]]
    total, count = 0.0, 0
    for idx, (h, p) in enumerate(zip(df["heard"], pts_list)):
        if pd.isna(h):
            continue
        key = (df["objects"].iloc[idx], int(h))
        if key in table:
            total += sum(p[i] for i in table[key]) / len(table[key])
            count += 1
    return float(total / max(count, 1))
