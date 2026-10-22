# name: maximal_vs_intermediate_competitor_preference
# description: On communicative trials where the heard word fits a minimal target, an intermediate competitor, and a maximal-feature competitor, the mean point difference between the maximal competitor and the intermediate competitor.
def test_statistic(df):
    table = {}
    distinct = df[["objects", "heard"]].drop_duplicates()
    for _, sc in distinct.iterrows():
        if pd.isna(sc["heard"]):
            continue
        h = int(sc["heard"])
        objs = json.loads(sc["objects"])
        n_feats = len(objs[0])
        max_idx = [i for i, o in enumerate(objs) if sum(o) == n_feats and o[h] == 1]
        inter_idx = [i for i, o in enumerate(objs) if 1 < sum(o) < n_feats and o[h] == 1]
        target_idx = [i for i, o in enumerate(objs) if sum(o) == 1 and o[h] == 1]
        if max_idx and inter_idx and target_idx:
            table[(sc["objects"], h)] = (max_idx[0], inter_idx)
    pts_list = [json.loads(p) if isinstance(p, str) else p for p in df["points"]]
    total, count = 0.0, 0
    for idx, (h, p) in enumerate(zip(df["heard"], pts_list)):
        if pd.isna(h):
            continue
        key = (df["objects"].iloc[idx], int(h))
        if key in table:
            m_idx, inter_indices = table[key]
            inter_mean = sum(p[i] for i in inter_indices) / len(inter_indices)
            total += (p[m_idx] - inter_mean)
            count += 1
    return float(total / max(count, 1))
