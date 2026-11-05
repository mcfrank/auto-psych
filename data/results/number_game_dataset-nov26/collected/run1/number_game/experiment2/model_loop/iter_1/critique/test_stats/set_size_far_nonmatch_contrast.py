# name: set_size_far_nonmatch_contrast
# description: Among 'background' targets (share no example's final digit and lie >=10 from every example), yes-rate in sets of 3-4 examples minus in sets of 1-2 examples; observed < null means people generalize more narrowly as examples accumulate (size-principle strictness) whereas the model's summed similarity grows with more examples, observed > null means broader.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    M = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        l = [int(v) for v in s.split(",")][:4]
        M[i, :len(l)] = l
    idx = pd.Series(np.arange(len(uniq)), index=uniq)[sets].to_numpy()
    E = M[idx]
    n = (~np.isnan(E)).sum(1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    D = np.where(np.isnan(E), np.inf, np.abs(E - t[:, None]))
    dm = ((np.nan_to_num(E, nan=-1) % 10) == (t % 10)[:, None]).any(1)
    bg = (~dm) & (D.min(1) >= 10)
    a = bg & (n >= 3)
    b = bg & (n <= 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
