# name: complex_competitor_minus_distractor_gap
# description: On communicative trials with both a feature-rich competitor (true of the heard message) and a feature-rich distractor (false of the heard message), the mean point difference between the competitor and the distractor.
def test_statistic(df):
    table = {}
    distinct = df[["objects", "heard"]].drop_duplicates()
    for _, sc in distinct.iterrows():
        if pd.isna(sc["heard"]):
            continue
        h = int(sc["heard"])
        objs = json.loads(sc["objects"])
        sat = [i for i, o in enumerate(objs) if o[h] == 1]
        unsat = [i for i, o in enumerate(objs) if o[h] == 0]
        fc = [sum(o) for o in objs]
        min_c = min(fc[i] for i in sat)
        comp_complex = [i for i in sat if fc[i] >= 3 and fc[i] > min_c]
        dist_complex = [i for i in unsat if fc[i] >= 3]
        if comp_complex and dist_complex:
            table[(sc["objects"], h)] = (comp_complex[0], dist_complex)
    pts_list = [json.loads(p) if isinstance(p, str) else p for p in df["points"]]
    total, count = 0.0, 0
    for idx, (h, p) in enumerate(zip(df["heard"], pts_list)):
        if pd.isna(h):
            continue
        key = (df["objects"].iloc[idx], int(h))
        if key in table:
            c_idx, d_indices = table[key]
            d_mean = sum(p[i] for i in d_indices) / len(d_indices)
            total += (p[c_idx] - d_mean)
            count += 1
    return float(total / max(count, 1))
