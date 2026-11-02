# name: arithmetic_relation_target_excess
# description: Yes rate for non-example targets equal to the sum or difference of two examples minus the yes rate of other non-example targets in the same sets (sets with 2+ examples); observed > null means people use arithmetic relations between examples that the model's concepts lack.
def test_statistic(df):
    sets = df["number_set"].unique()
    rel = {}
    exs = {}
    for s in sets:
        l = [int(x) for x in s.split(",")]
        exs[s] = set(l)
        r = set()
        for i in range(len(l)):
            for j in range(len(l)):
                if i != j:
                    r.add(l[i] + l[j]); r.add(abs(l[i] - l[j]))
        rel[s] = r
    items = df.groupby(["number_set", "target"])["rating"].mean().reset_index()
    multi = items["number_set"].map(lambda s: len(exs[s]) >= 2).to_numpy()
    is_rel = np.array([t in rel[s] for s, t in zip(items["number_set"], items["target"])])
    is_ex = np.array([t in exs[s] for s, t in zip(items["number_set"], items["target"])])
    a = items["rating"][multi & is_rel & ~is_ex]
    b = items["rating"][multi & ~is_rel & ~is_ex]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
