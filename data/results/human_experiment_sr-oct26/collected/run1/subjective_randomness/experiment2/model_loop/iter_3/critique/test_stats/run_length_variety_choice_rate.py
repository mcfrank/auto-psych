# name: run_length_variety_choice_rate
# description: Among trials where the two sequences have the same alternation rate but a different number of distinct run lengths, the proportion choosing the sequence with MORE distinct run lengths; observed above the null means people prize irregular run structure beyond what the model's alternation/imbalance/streak/symmetry terms capture, below means they prefer regular run structure more than the model predicts.
def test_statistic(df):
    def feats(s):
        runs = []; cur = 1
        for x, y in zip(s, s[1:]):
            if x == y:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        return len(runs), len(set(runs))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: feats(s) for s in u}
    na = df["sequence_a"].map(lambda s: m[s][0]).to_numpy(); nb = df["sequence_b"].map(lambda s: m[s][0]).to_numpy()
    va = df["sequence_a"].map(lambda s: m[s][1]).to_numpy(); vb = df["sequence_b"].map(lambda s: m[s][1]).to_numpy()
    sel = (na == nb) & (va != vb)
    if not sel.any():
        return 0.5
    chose_more = np.where(va > vb, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    return float(chose_more[sel].mean())
