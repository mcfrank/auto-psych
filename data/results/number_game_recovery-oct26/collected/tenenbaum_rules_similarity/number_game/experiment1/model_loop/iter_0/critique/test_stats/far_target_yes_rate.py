# name: far_target_yes_rate
# description: Proportion of "yes" for targets more than 40 away from every example; observed above the model's null means the model under-produces broad/random endorsement of distant targets (e.g. missing lapse or very broad concepts), below means it over-produces it.
def test_statistic(df):
    arr = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values[:, None]
    dist = np.nanmin(np.abs(arr - t), axis=1)
    m = dist > 40
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
