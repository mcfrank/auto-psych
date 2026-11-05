# name: sharpness_singleton_vs_large
# description: Std across targets of per-(set,target) yes-rates, averaged within single-example sets minus averaged within 3-4-example sets; observed > null means singleton generalization is more graded/sharp relative to large sets than the model's single slope across set sizes produces (i.e. model under-differentiates targets for singletons, or over-differentiates for large sets if lower).
def test_statistic(df):
    k = df["number_set"].astype(str).str.count(",") + 1
    cell = df.assign(k=k).groupby(["number_set", "target"]).agg(r=("rating", "mean"), k=("k", "first")).reset_index()
    sd = cell.groupby("number_set").agg(s=("r", "std"), k=("k", "first")).dropna()
    a = sd.loc[sd["k"] == 1, "s"]
    b = sd.loc[sd["k"] >= 3, "s"]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
