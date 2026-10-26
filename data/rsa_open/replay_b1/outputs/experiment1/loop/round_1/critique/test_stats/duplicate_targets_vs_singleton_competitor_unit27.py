# name: duplicate_targets_vs_singleton_competitor_unit27
# description: Total points on duplicate minimal targets [1, 1, 0] minus points on singleton competitor [1, 1, 1] when hearing feature 0.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if sc["heard"] == 0 and json.loads(sc["objects"]) == [[0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & (df["heard"] == sc["heard"])]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            diff = mat[:, [1, 2]].sum(axis=1) - mat[:, 3]
            return float(np.mean(diff))
    return 0.0
