# name: size_prior_max_feature_rate
# description: Rate of choosing the object with the maximal feature count on mumble trials in the size experiment.
def test_statistic(df):
    sub = df[(df["query"] == "prior") & (df["experiment"] == "size")]
    mapping = {}
    for obj_str in sub["objects"].unique():
        objs = json.loads(obj_str)
        counts = [sum(row) for row in objs]
        if max(counts) > min(counts):
            max_c = max(counts)
            mapping[obj_str] = counts.index(max_c)
    if not mapping:
        return 0.0
    matched = sub[sub["objects"].isin(mapping)]
    if len(matched) == 0:
        return 0.0
    target_idx = matched["objects"].map(mapping)
    return float(np.mean(matched["choice"] == target_idx))
