# name: chain_utterance_max_feature_competitor_choice_rate
# description: On utterance trials where all objects are unique singletons and every object satisfying the heard word has at least 2 features, the rate of choosing the true object with the maximum feature count.
def test_statistic(df):
    utt_df = df[df["heard"].notna()]
    if utt_df.empty:
        return 0.0
    pairs = utt_df[["objects", "heard"]].drop_duplicates()
    target_map = {}
    for obj_str, h_val in zip(pairs["objects"], pairs["heard"]):
        h = int(h_val)
        objs = np.asarray(json.loads(obj_str), dtype=int)
        diff = np.abs(objs[:, None, :] - objs[None, :, :]).sum(axis=2)
        is_sing = (diff == 0).sum(axis=1) == 1
        if not np.all(is_sing):
            continue
        truth = objs[:, h] == 1
        fcnt = objs.sum(axis=1)
        true_fcnts = fcnt[truth]
        if len(true_fcnts) >= 2 and true_fcnts.min() >= 2 and true_fcnts.max() > true_fcnts.min():
            max_f = true_fcnts.max()
            target_map[(obj_str, h)] = set(int(i) for i in np.where(truth & (fcnt == max_f))[0])
    if not target_map:
        return 0.0
    keys = list(zip(utt_df["objects"], utt_df["heard"].astype(int)))
    choices = utt_df["choice"].to_numpy()
    vals = [float(c in target_map[k]) for k, c in zip(keys, choices) if k in target_map]
    if not vals:
        return 0.0
    return float(np.mean(vals))
