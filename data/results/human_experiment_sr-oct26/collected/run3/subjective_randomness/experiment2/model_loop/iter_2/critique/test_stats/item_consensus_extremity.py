# name: item_consensus_extremity
# description: Mean over distinct (order-free) pairs of |proportion choosing the lexicographically smaller sequence - 0.5|; observed > null means the model under-produces item-level consensus (people agree more strongly on items than predicted), < null that it over-predicts how decisive items are.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    y = df["chose_left"].to_numpy().astype(bool)
    lo = np.where(a < b, a, b); hi = np.where(a < b, b, a)
    chosen = np.where(y, a, b)
    key = pd.Series(lo + "|" + hi)
    chose_lo = pd.Series((chosen == lo).astype(float))
    p = chose_lo.groupby(key).mean()
    return float((p - 0.5).abs().mean())
