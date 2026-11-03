# name: first_exposure_minus_repeat_yes_rate
# description: Per participant (trial_index restart), yes rate on the first trial on which each set appears minus the yes rate on that set's later trials; the model ignores exposure history, so t_observed far from null_mean means first encounters of a set are judged more (above) or less (below) generously than later ones.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[1, (np.diff(ti) < 0).astype(int)])
    tmp = pd.DataFrame({"pid": pid, "s": df["number_set"].astype(str).to_numpy(), "ti": ti, "r": df["rating"].to_numpy(dtype=float)})
    tmp = tmp.sort_values(["pid", "ti"], kind="mergesort")
    first = tmp.groupby(["pid", "s"]).cumcount().to_numpy() == 0
    r = tmp["r"].to_numpy()
    if first.sum() == 0 or (~first).sum() == 0:
        return 0.0
    return float(r[first].mean() - r[~first].mean())
