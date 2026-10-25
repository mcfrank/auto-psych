# name: asymmetric_5object_contrast_gap
# description: Difference in mean Likert rating between minimal target and duplicate competitor in 5-object asymmetric displays; tests under-production of pragmatic contrast in large scenes.
_CACHE = {}

def test_statistic(df):
    if "idx_min" not in _CACHE:
        unique = df[["objects", "heard", "probe"]].drop_duplicates()
        min_keys, comp_keys = set(), set()
        for _, r in unique.iterrows():
            objs = json.loads(r["objects"])
            if len(objs) == 5 and pd.notna(r["heard"]):
                h = int(r["heard"])
                if all(o[h] == 1 for o in objs):
                    p = int(r["probe"])
                    if sum(objs[p]) == 1:
                        min_keys.add((r["objects"], h, p))
                    elif sum(objs[p]) == 2:
                        comp_keys.add((r["objects"], h, p))
        keys = [(o, -1 if pd.isna(h) else int(h), p) for o, h, p in zip(df["objects"], df["heard"], df["probe"])]
        _CACHE["idx_min"] = np.where([k in min_keys for k in keys])[0]
        _CACHE["idx_comp"] = np.where([k in comp_keys for k in keys])[0]
    return float(df["rating"].iloc[_CACHE["idx_min"]].mean() - df["rating"].iloc[_CACHE["idx_comp"]].mean())
