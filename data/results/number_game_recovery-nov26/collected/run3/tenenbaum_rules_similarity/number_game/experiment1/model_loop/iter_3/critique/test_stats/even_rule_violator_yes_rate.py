# name: even_rule_violator_yes_rate
# description: Among multi-example sets whose examples are all multiples of some m in {4,6,8,12} (largest such m used), the yes-rate for non-example EVEN targets that are not multiples of m (e.g. 22 or 82 after "20,32,92"); observed below model means people reject parity-consistent numbers that break the specific multiples rule more firmly than the model (whose broad "even"/interval hypotheses leak probability to them), above means they accept them more.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    ex = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n_ex = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(int).to_numpy()
    m_chosen = np.zeros(len(df), dtype=int)
    for m in [4, 6, 8, 12]:
        allm = np.all(np.isnan(ex) | (np.nan_to_num(ex, nan=0.0) % m == 0), axis=1)
        m_chosen = np.where(allm, m, m_chosen)
    is_ex = np.any(ex == t[:, None], axis=1)
    safe_m = np.where(m_chosen > 0, m_chosen, 1)
    mask = (n_ex >= 2) & (m_chosen > 0) & (t % 2 == 0) & (t % safe_m != 0) & ~is_ex
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[mask].mean())
