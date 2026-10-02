# name: matching_endpoints_choice_rate
# description: Among trials where exactly one sequence starts and ends with the same flip (e.g. HTTTTTTH, THHHHHHT, HTHHTHHT), the proportion choosing that endpoint-matched sequence; observed below the null means people find a bracketed/framed sequence more designed than the model (which only penalises full palindromes) predicts, above means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ma = (a.str[0] == a.str[-1]).to_numpy(); mb = (b.str[0] == b.str[-1]).to_numpy()
    sel = ma != mb
    if not sel.any():
        return 0.5
    cl = df["chose_left"].to_numpy()
    chose_m = np.where(ma, cl, 1 - cl)
    return float(chose_m[sel].mean())
