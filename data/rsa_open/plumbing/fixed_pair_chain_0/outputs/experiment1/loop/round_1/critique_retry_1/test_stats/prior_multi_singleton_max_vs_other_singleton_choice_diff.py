# name: prior_multi_singleton_max_vs_other_singleton_choice_diff
# description: On prior trials (heard is NaN) with at least two unique singletons, the rate of choosing the singleton with the highest feature count minus the rate of choosing any lower-feature singleton.
def test_statistic(df):
    prior_df = df[df["heard"].isna()]
    if prior_df.empty:
        return 0.0
    unique_objs = prior_df["objects"].unique()
    max_sing_map = {}
    other_sing_map = {}
    valid_objs = set()
    for obj_str in unique_objs:
        objs = np.asarray(json.loads(obj_str), dtype=int)
        diff = np.abs(objs[:, None, :] - objs[None, :, :]).sum(axis=2)
        is_sing = (diff == 0).sum(axis=1) == 1
        if is_sing.sum() >= 2:
            sing_idx = np.where(is_sing)[0]
            fcnt = objs.sum(axis=1)
            max_f = fcnt[sing_idx].max()
            max_idx = int([i for i in sing_idx if fcnt[i] == max_f][0])
            others = set(int(i) for i in sing_idx if i != max_idx)
            valid_objs.add(obj_str)
            max_sing_map[obj_str] = max_idx
            other_sing_map[obj_str] = others
    sub = prior_df[prior_df["objects"].isin(valid_objs)]
    if sub.empty:
        return 0.0
    choices = sub["choice"].to_numpy()
    max_targets = sub["objects"].map(max_sing_map).to_numpy()
    chose_max = (choices == max_targets).astype(float)
    other_sets = sub["objects"].map(other_sing_map).to_numpy()
    chose_other = np.array([float(c in s) for c, s in zip(choices, other_sets)], dtype=float)
    return float(chose_max.mean() - chose_other.mean())
