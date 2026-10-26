# name: multiclone_competitor_allocation_unit20
# description: Mean points allocated to the singleton competitor [1, 1, 1] in contexts with three identical clones of minimal target [0, 0, 1] when hearing feature 2.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if sc["heard"] == 2 and json.loads(sc["objects"]) == [[0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & (df["heard"] == sc["heard"])]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            return float(np.mean(mat[:, 3]))
    return 0.0
