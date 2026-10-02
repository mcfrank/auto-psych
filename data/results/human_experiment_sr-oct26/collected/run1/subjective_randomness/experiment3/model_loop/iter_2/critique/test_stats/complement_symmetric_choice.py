# name: complement_symmetric_choice
# description: Among trials where exactly one sequence (length >= 4) equals its own reverse with H/T swapped and is NOT strictly alternating, the proportion choosing that antisymmetric sequence; observed below null_mean means people reject non-alternating antisymmetric sequences even more than the model's single shared antisymmetry weight predicts (above: the weight over-penalises them, e.g. it is driven by strict alternation).
def test_statistic(df):
    def anti(s):
        u = pd.unique(s)
        def f(x):
            if len(x) < 4:
                return 0
            alt = all(x[i] != x[i + 1] for i in range(len(x) - 1))
            fl = "".join("T" if ch == "H" else "H" for ch in x[::-1])
            return int(x == fl and not alt)
        return s.map({x: f(x) for x in u}).to_numpy()
    aa, ab = anti(df["sequence_a"]), anti(df["sequence_b"])
    mask = aa != ab
    if not mask.any():
        return 0.5
    c = df["chose_left"].to_numpy()
    chose_anti = np.where(aa == 1, c, 1 - c)
    return float(chose_anti[mask].mean())
