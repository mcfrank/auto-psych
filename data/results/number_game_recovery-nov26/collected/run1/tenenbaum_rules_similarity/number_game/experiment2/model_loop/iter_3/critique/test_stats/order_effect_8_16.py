# name: order_effect_8_16
# description: Yes rate for set "16,8" minus yes rate for set "8,16" over the targets both orders were asked about; the model is order-invariant (null ~0), so a nonzero observed value means presentation order changes generalisation (positive = descending order generalises more).
def test_statistic(df):
    a = df[df["number_set"] == "16,8"]
    b = df[df["number_set"] == "8,16"]
    common = set(a["target"]) & set(b["target"])
    if not common:
        return 0.0
    a = a[a["target"].isin(common)]
    b = b[b["target"].isin(common)]
    return float(a["rating"].mean() - b["rating"].mean())
