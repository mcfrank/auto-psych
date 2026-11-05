# name: background_trial_drift_corr
# description: Among 'background' targets (share no example's final digit and lie >=10 from every example), Pearson correlation between trial_index and the yes response; observed > null means the low-similarity yes floor rises over the session (growing guessing/lapse or loosening), observed < null means it falls, which the stationary model cannot produce.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    M = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        l = [int(v) for v in s.split(",")][:4]
        M[i, :len(l)] = l
    idx = pd.Series(np.arange(len(uniq)), index=uniq)[sets].to_numpy()
    E = M[idx]
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    D = np.where(np.isnan(E), np.inf, np.abs(E - t[:, None]))
    dm = ((np.nan_to_num(E, nan=-1) % 10) == (t % 10)[:, None]).any(1)
    bg = (~dm) & (D.min(1) >= 10)
    x = df["trial_index"].astype(float).to_numpy()[bg]
    y = r[bg]
    if len(y) < 3 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
