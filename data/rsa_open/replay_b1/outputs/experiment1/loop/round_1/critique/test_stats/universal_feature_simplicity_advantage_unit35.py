# name: universal_feature_simplicity_advantage_unit35
# description: Mean points on minimal 1-feature objects [0, 0, 1] minus mean points on extra-feature objects [0, 1, 1] and [1, 0, 1] when heard feature 2 applies to all objects.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if sc["heard"] == 2 and json.loads(sc["objects"]) == [[0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & (df["heard"] == sc["heard"])]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            diff = mat[:, [0, 1]].mean(axis=1) - mat[:, [2, 3]].mean(axis=1)
            return float(np.mean(diff))
    return 0.0
