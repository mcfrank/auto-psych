# name: split_half_agreement_correlation
# description: Across participants, Pearson correlation between majority-agreement rate on odd-numbered trials and on even-numbered trials; observed above null_mean means agreement with the consensus is a more stable personal trait than the model's person-level parameters imply, below means less stable.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    sw = a < b
    key = np.char.add(np.char.add(np.where(sw, a, b), "|"), np.where(sw, b, a))
    cl = df["chose_left"].to_numpy().astype(float)
    chose_first = np.where(sw, cl, 1 - cl)
    pair_rate = pd.Series(chose_first).groupby(key).transform("mean").to_numpy()
    agree = (chose_first == (pair_rate >= 0.5)).astype(float)
    odd = (df["trial_index"].to_numpy() % 2).astype(int)
    t = pd.DataFrame({"p": df["participant_id"].to_numpy(), "o": odd, "g": agree}).groupby(["p", "o"])["g"].mean().unstack().dropna()
    if len(t) < 3:
        return 0.0
    r = np.corrcoef(t[0].to_numpy(), t[1].to_numpy())[0, 1]
    return float(r) if np.isfinite(r) else 0.0
