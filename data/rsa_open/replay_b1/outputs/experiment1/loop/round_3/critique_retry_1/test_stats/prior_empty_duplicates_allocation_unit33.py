# name: prior_empty_duplicates_allocation_unit33
# description: Total points allocated to duplicate empty objects [0, 0, 0] minus points allocated to 3-feature singleton [1, 1, 1] on uninformative prior trials.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if pd.isna(sc["heard"]) and json.loads(sc["objects"]) == [[0, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & df["heard"].isna()]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            diff = mat[:, [0, 1]].sum(axis=1) - mat[:, 3]
            return float(np.mean(diff))
    return 0.0
