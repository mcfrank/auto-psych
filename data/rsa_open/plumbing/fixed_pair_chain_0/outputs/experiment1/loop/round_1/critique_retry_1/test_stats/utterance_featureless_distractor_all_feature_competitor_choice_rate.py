# name: utterance_featureless_distractor_all_feature_competitor_choice_rate
# description: On utterance trials containing at least one featureless object (all 0s), at least one 1-feature true object, and a unique all-feature object (all 1s), the rate of choosing the all-feature object.
def test_statistic(df):
    utt_df = df[df["heard"].notna()]
    if utt_df.empty:
        return 0.0
    pairs = utt_df[["objects", "heard"]].drop_duplicates()
    target_map = {}
    for obj_str, h_val in zip(pairs["objects"], pairs["heard"]):
        h = int(h_val)
        objs = np.asarray(json.loads(obj_str), dtype=int)
        n_feat = objs.shape[1]
        truth = objs[:, h] == 1
        fcnt = objs.sum(axis=1)
        has_zero = np.any(fcnt == 0)
        has_one_true = np.any(truth & (fcnt == 1))
        all_feat_idx = np.where(truth & (fcnt == n_feat))[0]
        if has_zero and has_one_true and len(all_feat_idx) == 1:
            target_map[(obj_str, h)] = int(all_feat_idx[0])
    if not target_map:
        return 0.0
    keys = list(zip(utt_df["objects"], utt_df["heard"].astype(int)))
    choices = utt_df["choice"].to_numpy()
    vals = [float(c == target_map[k]) for k, c in zip(keys, choices) if k in target_map]
    if not vals:
        return 0.0
    return float(np.mean(vals))
