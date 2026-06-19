# name: na_length_choice_rate_gap
# description: Mean chose_left for n_a = 2 (short A) minus mean chose_left for n_a >= 4 (longer A); tests whether the model captures the full magnitude of the sequence-length contrast in left-choice rates driven by differences in A's apparent non-randomness.
def test_statistic(df):
    short = df[df["n_a"] == 2]["chose_left"].mean()
    long_ = df[df["n_a"] >= 4]["chose_left"].mean()
    return float(short - long_)
