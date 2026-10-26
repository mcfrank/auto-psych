# name: multiclone_competitor_margin_unit20
# description: Points allocated to singleton competitor [1, 1, 1] minus mean points allocated to three identical clones [0, 0, 1] when hearing feature 2.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if sc["heard"] == 2 and json.loads(sc["objects"]) == [[0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & (df["heard"] == sc["heard"])]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            diff = mat[:, 3] - mat[:, [0, 1, 2]].mean(axis=1)
            return float(np.mean(diff))
    return 0.0
