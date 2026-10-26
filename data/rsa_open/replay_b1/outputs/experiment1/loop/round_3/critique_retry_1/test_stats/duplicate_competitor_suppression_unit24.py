# name: duplicate_competitor_suppression_unit24
# description: Total points allocated to duplicate competitors [1, 1, 1] when hearing feature 2 in scenes with a minimal target [0, 0, 1] and an empty distractor [0, 0, 0].
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if sc["heard"] == 2 and json.loads(sc["objects"]) == [[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & (df["heard"] == sc["heard"])]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            return float(np.mean(mat[:, [2, 3]].sum(axis=1)))
    return 0.0
