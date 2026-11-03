# name: digit_reversal_yes_rate
# description: Yes rate among trials whose target is the digit-reversal of a two-digit, non-repdigit example (and not itself an example); observed above the model's null mean means people treat reversed numerals as same-program members more than the rule/interval model predicts.
def test_statistic(df):
    t = df["target"].astype(int)
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    ex = ex.reindex(columns=range(4))
    tv = t.values
    rev_hit = np.zeros(len(df), dtype=bool)
    is_ex = np.zeros(len(df), dtype=bool)
    for c in range(4):
        x = ex[c].values
        ok = ~np.isnan(x)
        xi = np.where(ok, x, 0).astype(int)
        r = (xi % 10) * 10 + xi // 10
        valid = ok & (xi >= 10) & (xi <= 99) & (xi % 11 != 0) & (xi % 10 != 0)
        rev_hit |= valid & (r == tv)
        is_ex |= ok & (xi == tv)
    m = rev_hit & ~is_ex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
