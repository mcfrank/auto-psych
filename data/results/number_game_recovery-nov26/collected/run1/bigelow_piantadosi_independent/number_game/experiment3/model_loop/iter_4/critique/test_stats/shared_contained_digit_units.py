# name: shared_contained_digit_units
# description: In 2-4 example sets where every example contains a common digit d but not all end in it, among non-example targets more than 5 from every example, yes-rate for targets ending in such a d minus targets containing no such d; observed above the null means people extend a shared contained digit into the units place more than the model's contains/ends-in rules predict, below means the model over-produces it.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    Ei = np.nan_to_num(E, nan=-1).astype(int)
    common = np.zeros((len(t), 10), dtype=bool)
    endsall = np.zeros((len(t), 10), dtype=bool)
    tcont = np.zeros((len(t), 10), dtype=bool)
    for d in range(10):
        has = (np.mod(Ei, 10) == d) | (((Ei // 10) % 10 == d) & (Ei >= 10)) | ((Ei == 100) & (d in (0, 1)))
        common[:, d] = np.all(has | ~valid, axis=1)
        endsall[:, d] = np.all((np.mod(Ei, 10) == d) | ~valid, axis=1)
        tcont[:, d] = (t % 10 == d) | (((t // 10) % 10 == d) & (t >= 10)) | ((t == 100) & (d in (0, 1)))
    cd = common & ~endsall
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    m = cd.any(axis=1) & (n >= 2) & (~is_ex) & (dist > 5)
    unit_hit = cd[np.arange(len(t)), t % 10]
    none = ~np.any(cd & tcont, axis=1)
    a, b = m & unit_hit, m & none
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
