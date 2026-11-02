# name: far_target_yes_rate
# description: Yes-rate for targets lying >= 25 away from every example; observed > null means people over-generalise to distant numbers (the model's rules/size principle under-produce far "yes"), observed < null means the model over-generalises far away.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(float).to_numpy()[:, None]
    dmin = np.nanmin(np.abs(ex - t), axis=1)
    mask = dmin >= 25
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[mask].mean())
