# name: redundant_multifeature_competitor_rating
# description: Mean Likert rating of a probed scalar competitor with redundant features when a minimal referent exists; tests lack of feature parsimony on informative trials.
_CACHE = {}

def test_statistic(df):
    if "idx" not in _CACHE:
        unique = df[["objects", "heard", "probe"]].drop_duplicates()
        matching = set()
        for _, r in unique.iterrows():
            if pd.notna(r["heard"]):
                objs = json.loads(r["objects"])
                h = int(r["heard"])
                p = int(r["probe"])
                n_feats = len(objs[0])
                if objs[p][h] == 1:
                    p_fc = sum(objs[p])
                    if 2 <= p_fc < n_feats:
                        if any(o[h] == 1 and sum(o) < p_fc for i, o in enumerate(objs) if i != p):
                            if not all(o[h] == 1 for o in objs):
                                matching.add((r["objects"], h, p))
        keys = [(o, -1 if pd.isna(h) else int(h), p) for o, h, p in zip(df["objects"], df["heard"], df["probe"])]
        _CACHE["idx"] = np.where([k in matching for k in keys])[0]
    return float(df["rating"].iloc[_CACHE["idx"]].mean())
