# name: final_run_avoidance_equal_switches
# description: Among trials whose sequences have equal switch counts but different final-run lengths (length of the streak at the end), the proportion choosing the sequence with the shorter final run; observed > null means the model under-weights avoidance of a streak at the end (recency/position effect beyond switch count), < null the reverse.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    def switches(s):
        n = s.str.len().max()
        p = s.str.pad(n, side="right", fillchar=".")
        tot = np.zeros(len(s))
        for i in range(n - 1):
            c1 = p.str[i].to_numpy(); c2 = p.str[i + 1].to_numpy()
            tot += ((c1 != c2) & (c1 != ".") & (c2 != ".")).astype(float)
        return tot
    def final_run(s):
        r = s.str[::-1]
        first = r.str[0]
        lenH = r.str.extract(r"^(H*)")[0].str.len().to_numpy()
        lenT = r.str.extract(r"^(T*)")[0].str.len().to_numpy()
        return np.where(first.to_numpy() == "H", lenH, lenT).astype(float)
    ka = switches(a); kb = switches(b)
    fa = final_run(a); fb = final_run(b)
    m = (ka == kb) & (fa != fb)
    y = df["chose_left"].to_numpy()
    chose_short = np.where(fa < fb, y, 1 - y)[m]
    return float(chose_short.mean())
