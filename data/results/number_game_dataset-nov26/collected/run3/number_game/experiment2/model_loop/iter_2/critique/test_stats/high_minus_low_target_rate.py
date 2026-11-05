# name: high_minus_low_target_rate
# description: Across all sets, non-example-target yes rate for targets > 50 minus targets <= 50; observed above null means people favour large numbers more than the model (a magnitude bias the model lacks); below = model over-favours large targets.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    E = pd.DataFrame(ex.tolist()).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    isex = (E == t[:, None]).any(axis=1)
    mn = np.nanmin(E, axis=1); mx = np.nanmax(E, axis=1)
    a = (~isex) & (t > 50); b = (~isex) & (t <= 50)
    return float(r[a].mean() - r[b].mean())
