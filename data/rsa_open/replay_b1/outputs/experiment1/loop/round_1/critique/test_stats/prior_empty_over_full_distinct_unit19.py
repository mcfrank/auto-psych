# name: prior_empty_over_full_distinct_unit19
# description: Points allocated to empty object [0, 0, 0] minus points allocated to full object [1, 1, 1] on prior trials where all four objects are distinct.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if pd.isna(sc["heard"]) and json.loads(sc["objects"]) == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & df["heard"].isna()]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            diff = mat[:, 0] - mat[:, 3]
            return float(np.mean(diff))
    return 0.0
