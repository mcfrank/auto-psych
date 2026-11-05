# name: ruleless_set_in_range_rate
# description: On 2+ example sets that no rule (odd/even, multiples 3-12, powers, squares, cubes, primes, ends-in/contains digit) fits fully, yes rate for in-range non-example targets; observed above null means the model under-endorses the range when no rule explains every example (too much belief kept on near-fitting rules).
import numpy as _np
def _rules():
    N = _np.arange(1, 101)
    R = [N % 2 == 1, N % 2 == 0]
    R.append(_np.array([n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1)) for n in N]))
    R.append(_np.isin(N, [k * k for k in range(1, 11)]))
    R.append(_np.isin(N, [k ** 3 for k in range(1, 5)]))
    R.append(_np.isin(N, [2, 4, 8, 16, 64]))
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        R.append(N % m == 0)
    for m in range(2, 11):
        R.append(_np.isin(N, [m ** k for k in range(1, 8) if m ** k <= 100]))
    for dg in range(10):
        R.append(N % 10 == dg)
        R.append(_np.array([str(dg) in str(n) for n in N]))
    return _np.array(R)
_R = _rules()
def test_statistic(df):
    sets = df["number_set"].unique()
    ruleless = {}
    for s in sets:
        ex = _np.array([int(x) for x in s.split(",")]) - 1
        ruleless[s] = (len(ex) >= 2) and not bool(_R[:, ex].all(axis=1).any())
    rl = df["number_set"].map(ruleless).astype(bool)
    sp = df["number_set"].str.split(",", expand=True)
    v = sp.apply(pd.to_numeric, errors="coerce")
    lo, hi = v.min(axis=1), v.max(axis=1)
    t = df["target"].astype(int)
    isex = v.eq(t, axis=0).any(axis=1)
    sel = rl & (~isex) & (t > lo) & (t < hi)
    if sel.sum() == 0:
        return 0.0
    return float(df.loc[sel, "rating"].mean())
