# name: maximal_referent_duplicate_distractors_rating
# description: Mean Likert rating of a probed maximal referent when distractors sharing the heard word are duplicated minimal objects; tests under-prediction of maximal referent in duplicate scenes.
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
                if n_feats >= 3 and len(objs) <= 4:
                    if sum(objs[p]) == n_feats:
                        other_with_h = [o for i, o in enumerate(objs) if i != p and o[h] == 1]
                        if other_with_h and all(sum(o) == 1 for o in other_with_h):
                            if len(other_with_h) >= 2 and any(other_with_h.count(o) >= 2 for o in other_with_h):
                                matching.add((r["objects"], h, p))
        keys = [(o, -1 if pd.isna(h) else int(h), p) for o, h, p in zip(df["objects"], df["heard"], df["probe"])]
        _CACHE["idx"] = np.where([k in matching for k in keys])[0]
    return float(df["rating"].iloc[_CACHE["idx"]].mean())
