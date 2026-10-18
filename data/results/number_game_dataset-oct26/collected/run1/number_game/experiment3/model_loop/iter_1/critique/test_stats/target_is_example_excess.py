# name: target_is_example_excess
# description: Yes-rate for targets identical to one of the shown examples minus yes-rate for other targets that share an example's final digit; observed > null means the model under-credits exact repeats of an example (it only sees the shared final digit).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    t = df["target"].astype(int)
    lists = ns.str.split(",")
    uniq = pd.Series(ns.unique())
    sets_map = {s: set(int(x) for x in s.split(",")) for s in uniq}
    digs_map = {s: set(int(x) % 10 for x in s.split(",")) for s in uniq}
    key = ns + "|" + t.astype(str)
    pairs = pd.DataFrame({"ns": ns, "t": t}).drop_duplicates()
    is_ex = {f"{a}|{b}": (b in sets_map[a]) for a, b in zip(pairs["ns"], pairs["t"])}
    dig = {f"{a}|{b}": ((b % 10) in digs_map[a]) for a, b in zip(pairs["ns"], pairs["t"])}
    ex = key.map(is_ex).astype(bool)
    dm = key.map(dig).astype(bool)
    r = df["rating"].astype(float)
    a = r[ex]
    b = r[dm & ~ex]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
