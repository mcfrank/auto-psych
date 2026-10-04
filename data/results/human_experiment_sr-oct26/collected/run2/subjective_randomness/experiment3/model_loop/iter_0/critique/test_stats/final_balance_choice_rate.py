# name: final_balance_choice_rate
# description: Among pairs differing in final |#H - #T| imbalance, the rate the more balanced sequence is chosen; observed above null means people reward equal heads/tails counts more than the model's tally-span term captures (below null: less).
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ia = (2 * a.str.count("H") - a.str.len()).abs().to_numpy()
    ib = (2 * b.str.count("H") - b.str.len()).abs().to_numpy()
    m = ia != ib
    if m.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where((ia < ib)[m], c, 1 - c)))
