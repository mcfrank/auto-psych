# name: decade_shared_contrast
# description: Yes rate for non-example targets sharing a tens digit with some example minus yes rate for targets sharing none (among sets of >=2 examples); observed below null means the model over-weights local magnitude neighbourhoods relative to people.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",")
    t = df["target"].astype(int).values
    mat = np.full((len(df), 4), -1)
    for j in range(4):
        mat[:, j] = pd.to_numeric(s.str[j], errors="coerce").fillna(-1).astype(int).values
    valid = mat > 0
    n = valid.sum(1)
    shown = np.any(valid & (mat == t[:, None]), 1)
    dec = np.any(valid & (mat // 10 == (t // 10)[:, None]), 1)
    r = df["rating"].values
    a = (n >= 2) & ~shown & dec; b = (n >= 2) & ~dec
    return float(r[a].mean() - r[b].mean())
