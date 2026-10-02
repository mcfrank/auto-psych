# name: split_half_side_bias_correlation
# description: Across participants, Pearson correlation between the Left-choice rate on odd-numbered trials and on even-numbered trials; observed above null_mean means left/right preference is a stable personal trait the model (no side term) does not produce, near null means apparent side-bias spread is just stimulus noise.
def test_statistic(df):
    odd = (df["trial_index"].to_numpy() % 2).astype(int)
    t = pd.DataFrame({"p": df["participant_id"].to_numpy(), "o": odd, "c": df["chose_left"].to_numpy().astype(float)}).groupby(["p", "o"])["c"].mean().unstack().dropna()
    if len(t) < 3:
        return 0.0
    r = np.corrcoef(t[0].to_numpy(), t[1].to_numpy())[0, 1]
    return float(r) if np.isfinite(r) else 0.0
