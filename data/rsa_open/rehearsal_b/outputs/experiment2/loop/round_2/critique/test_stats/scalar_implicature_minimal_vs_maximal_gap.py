# name: scalar_implicature_minimal_vs_maximal_gap
# description: Difference in mean Likert rating between minimal referent and maximal competitor in 4-object scalar implicature scenes; tests over-polarization of scalar implicatures.
_CACHE = {}

def test_statistic(df):
    if "idx_min" not in _CACHE:
        unique = df[["objects", "heard", "probe"]].drop_duplicates()
        min_keys, max_keys = set(), set()
        for _, r in unique.iterrows():
            if pd.notna(r["heard"]):
                objs = json.loads(r["objects"])
                h = int(r["heard"])
                p = int(r["probe"])
                if len(objs) == 4 and objs[p][h] == 1:
                    comp_objs = [o for o in objs if o[h] == 1]
                    n_feats = len(objs[0])
                    has_min = any(sum(o) == 1 and o[h] == 1 for o in comp_objs)
                    has_max = any(sum(o) == n_feats for o in comp_objs)
                    if has_min and has_max and len(comp_objs) == 2:
                        if sum(objs[p]) == 1:
                            min_keys.add((r["objects"], h, p))
                        elif sum(objs[p]) == n_feats:
                            max_keys.add((r["objects"], h, p))
        keys = [(o, -1 if pd.isna(h) else int(h), p) for o, h, p in zip(df["objects"], df["heard"], df["probe"])]
        _CACHE["idx_min"] = np.where([k in min_keys for k in keys])[0]
        _CACHE["idx_max"] = np.where([k in max_keys for k in keys])[0]
    return float(df["rating"].iloc[_CACHE["idx_min"]].mean() - df["rating"].iloc[_CACHE["idx_max"]].mean())
