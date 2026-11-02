# name: single_example_yes_rate
# description: Yes rate on trials whose set has exactly one example (target not equal to it); observed < null means the model generalises too broadly from one example, observed > null too narrowly.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    first = pd.to_numeric(df["number_set"].astype(str).str.split(",").str[0], errors="coerce")
    m = (n == 1).values & (first.values != df["target"].astype(float).values)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
