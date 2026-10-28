# name: within_set_cell_spread
# description: For sets asked about with >= 5 distinct targets, the standard deviation of per-target yes-rates within the set, averaged across sets; observed > null means people's generalization within a set is sharper / more all-or-none across targets than the model predicts (observed < null: flatter, more graded than the model).
def test_statistic(df):
    cell = df.groupby(["number_set", "target"])["rating"].mean().reset_index()
    nt = cell.groupby("number_set")["target"].transform("size")
    cell = cell[nt >= 5]
    if len(cell) == 0:
        return 0.0
    sd = cell.groupby("number_set")["rating"].std(ddof=0)
    return float(sd.mean())
