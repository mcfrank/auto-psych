# name: shared_digit_containment_effect
# description: For 2-4 example sets whose examples all contain some digit d but do not all share a last digit (e.g. 8,80,48; 78,48,88 -> 8), yes rate on non-example targets containing d but whose last digit matches no example, minus targets lacking d and matching no example's last digit; observed above null_mean means people generalize by "contains digit d" more than the model predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    nanm = np.isnan(ex)
    exz = np.where(nanm, 0.0, ex)
    ones, tens = np.mod(exz, 10), np.floor(exz / 10)
    ld = np.where(nanm, np.nan, ones)
    same = np.all((ld == ld[:, :1]) | nanm, axis=1)
    tone, tten = np.mod(t, 10), np.floor(t / 10)
    a = np.zeros(len(t), bool); b = np.zeros(len(t), bool)
    base = (n >= 2) & ~same & ~np.any(ex == t[:, None], axis=1) & ~np.any(ld == tone[:, None], axis=1)
    done = np.zeros(len(t), bool)
    for d in range(1, 10):
        has = np.all(((ones == d) | (tens == d)) | nanm, axis=1) & ~done
        tc = (tone == d) | (tten == d)
        a |= base & has & tc
        b |= base & has & ~tc
        done |= has
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
