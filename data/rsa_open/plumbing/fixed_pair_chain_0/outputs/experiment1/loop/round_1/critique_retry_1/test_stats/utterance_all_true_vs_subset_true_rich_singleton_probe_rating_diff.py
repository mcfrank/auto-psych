# name: utterance_all_true_vs_subset_true_rich_singleton_probe_rating_diff
# description: On utterance trials where second_probe is a true singleton with at least 3 features, mean second_rating when the word is true of all objects (with one singleton) minus when it is true of a proper subset including a 1-feature object.
def test_statistic(df):
    utt_df = df[df["heard"].notna()]
    if utt_df.empty:
        return 0.0
    triples = utt_df[["objects", "heard", "second_probe"]].drop_duplicates()
    group_map = {}
    for obj_str, h_val, prb_val in zip(triples["objects"], triples["heard"], triples["second_probe"]):
        h = int(h_val)
        prb = int(prb_val)
        objs = np.asarray(json.loads(obj_str), dtype=int)
        n_obj = len(objs)
        truth = objs[:, h] == 1
        diff = np.abs(objs[:, None, :] - objs[None, :, :]).sum(axis=2)
        is_sing = (diff == 0).sum(axis=1) == 1
        fcnt = objs.sum(axis=1)
        if is_sing[prb] and truth[prb] and fcnt[prb] >= 3:
            if truth.sum() == n_obj and is_sing.sum() == 1:
                group_map[(obj_str, h, prb)] = 1
            elif truth.sum() < n_obj and np.any(truth & (fcnt == 1)):
                group_map[(obj_str, h, prb)] = -1
    if not group_map:
        return 0.0
    keys = list(zip(utt_df["objects"], utt_df["heard"].astype(int), utt_df["second_probe"].astype(int)))
    grp = np.array([group_map.get(k, 0) for k in keys], dtype=int)
    ratings = utt_df["second_rating"].to_numpy().astype(float)
    m1 = grp == 1
    m2 = grp == -1
    if not np.any(m1) or not np.any(m2):
        return 0.0
    return float(ratings[m1].mean() - ratings[m2].mean())
