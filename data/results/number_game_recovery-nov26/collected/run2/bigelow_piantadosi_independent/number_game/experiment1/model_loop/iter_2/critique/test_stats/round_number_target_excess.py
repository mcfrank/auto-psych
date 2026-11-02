# name: round_number_target_excess
# description: In sets whose examples do not all end in 0, yes rate for non-example targets that are multiples of 10 minus other non-example targets at the same nearest-example distance band (>5 away); t_observed above null_mean means people over-endorse round numbers relative to the model, below means they avoid them more.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    v = ~np.isnan(s)
    t = df["target"].astype(int).to_numpy()
    tf = t.astype(float)
    all0 = ((np.where(v, s % 10, 0) == 0)).all(axis=1)
    dist = np.nanmin(np.abs(s - tf[:, None]), axis=1)
    r = df["rating"].to_numpy(dtype=float)
    base = ~all0 & (dist > 5)
    a = base & (t % 10 == 0); b = base & (t % 10 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
