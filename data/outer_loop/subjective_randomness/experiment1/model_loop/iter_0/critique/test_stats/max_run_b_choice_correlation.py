# name: max_run_b_choice_correlation
# description: Pearson correlation between max_run_norm_b and chose_left; tests whether longer B runs monotonically increase the probability of choosing A, probing max-run as a salience cue absent from the model.
def test_statistic(df):
    r = np.corrcoef(df['max_run_norm_b'].values, df['chose_left'].values)[0, 1]
    return float(r)
