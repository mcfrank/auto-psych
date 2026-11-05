# name: shared_final_digit_match_rate
# description: Yes-rate for targets that end in the final digit shared by all (>=2) examples (e.g. 28,98 -> 38, 58); observed > null means the model under-endorses targets fitting a shared last-digit rule (treats it as just one diluted property), observed < null means it over-endorses them.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    m = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        d = {e % 10 for e in ex}
        m[s] = d.pop() if (len(ex) >= 2 and len(d) == 1) else -1
    fd = sets.map(m).to_numpy()
    t = df["target"].astype(int).to_numpy() % 10
    sel = (fd >= 0) & (t == fd)
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).to_numpy()[sel].mean())
