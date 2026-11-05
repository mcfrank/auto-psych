# name: odd_set_even_target_rate
# description: Yes-rate for even targets when all (>=2) examples are odd; observed > null means the model's single parity salience makes people look stricter about odd sets than they are (people tolerate parity violations of odd sets), observed < null the reverse.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    m = {}
    for s in uniq:
        ex = [int(x) for x in s.split(",")]
        m[s] = len(ex) >= 2 and all(e % 2 == 1 for e in ex)
    allodd = sets.map(m).to_numpy().astype(bool)
    even = (df["target"].astype(int).to_numpy() % 2) == 0
    sel = allodd & even
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).to_numpy()[sel].mean())
