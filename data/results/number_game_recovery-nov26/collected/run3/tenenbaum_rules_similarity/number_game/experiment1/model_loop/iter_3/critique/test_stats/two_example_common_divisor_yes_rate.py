# name: two_example_common_divisor_yes_rate
# description: Among 2-example sets whose examples share a common divisor >= 3 (or a last digit), the yes-rate for non-example targets that also share that divisor (gcd(gcd of examples, target) >= 3) or that last digit; observed above model means people commit to the specific arithmetic rule from just two examples more sharply than the model's size principle + interval competition predicts, below means less.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    n_ex = parts.notna().sum(axis=1)
    two = (n_ex == 2).to_numpy()
    a = pd.to_numeric(parts[0], errors="coerce").fillna(1).astype(int).to_numpy()
    b = pd.to_numeric(parts[1], errors="coerce").fillna(1).astype(int).to_numpy() if parts.shape[1] > 1 else a
    t = df["target"].astype(int).to_numpy()
    g = np.gcd(a, b)
    shared_div = np.gcd(g, t) >= 3
    same_ld = (a % 10 == b % 10) & (t % 10 == a % 10)
    eligible = two & ((g >= 3) | (a % 10 == b % 10)) & (t != a) & (t != b)
    mask = eligible & (shared_div | same_ld)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[mask].mean())
