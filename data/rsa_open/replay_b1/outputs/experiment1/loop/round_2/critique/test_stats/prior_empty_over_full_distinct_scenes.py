# name: prior_empty_over_full_distinct_scenes
# description: Mean points on empty object [0, 0, 0] minus mean points on full object [1, 1, 1] on uninformative prior trials across 4-object displays where all objects are distinct.
def test_statistic(df):
    target_scenes = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
    ]
    diffs = []
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if pd.isna(sc["heard"]):
            objs = json.loads(sc["objects"])
            if objs in target_scenes:
                sub = df[(df["objects"] == sc["objects"]) & df["heard"].isna()]
                mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
                diffs.append(mat[:, 0] - mat[:, 3])
    if diffs:
        return float(np.mean(np.concatenate(diffs)))
    return 0.0
