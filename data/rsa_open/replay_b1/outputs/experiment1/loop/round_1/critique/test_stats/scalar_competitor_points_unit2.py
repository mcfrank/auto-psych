# name: scalar_competitor_points_unit2
# description: Mean points allocated to the overspecified competitor [1, 1, 1] when hearing feature 2 in the 4-object scalar scene with duplicate foils [1, 1, 0].
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if sc["heard"] == 2 and json.loads(sc["objects"]) == [[0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & (df["heard"] == sc["heard"])]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            return float(np.mean(mat[:, 3]))
    return 0.0
