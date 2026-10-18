# name: singleton_same_digit_yes
# description: Yes rate on 1-example sets for targets ending in the example's final digit (target != example); observed below null means the model over-generalizes from a single example's last digit (single-example digit matches are weaker than on multi-example sets), above means people endorse them even more.
def test_statistic(df):
    s = df["number_set"].astype(str)
    n = s.str.count(",") + 1
    t = df["target"].astype(int)
    m = n == 1
    x = pd.to_numeric(s.where(m, "0"), errors="coerce").fillna(0).astype(int)
    sel = m & ((x % 10) == (t % 10)) & (x != t)
    if sel.sum() == 0:
        return 0.0
    return float(df.loc[sel, "rating"].mean())
