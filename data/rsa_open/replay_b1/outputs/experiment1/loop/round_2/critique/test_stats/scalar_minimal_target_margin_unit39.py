# name: scalar_minimal_target_margin_unit39
# description: Points allocated to minimal target [0, 0, 1] minus points allocated to 2-feature competitor [0, 1, 1] when hearing feature 2 in scenes with duplicate 3-feature competitors.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if sc["heard"] == 2 and json.loads(sc["objects"]) == [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & (df["heard"] == sc["heard"])]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            diff = mat[:, 0] - mat[:, 1]
            return float(np.mean(diff))
    return 0.0
