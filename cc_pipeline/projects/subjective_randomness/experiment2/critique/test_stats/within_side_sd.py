import math


def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)

    Average within-group standard deviation of chose_left_pct, where groups are:
    - "alt_left": stimuli where the alternating sequence is sequence_a (left)
    - "alt_right": stimuli where the alternating sequence is sequence_b (right)

    Within each group, all models that score only on alternation rate or balance
    predict an identical P(chose_left), so within-group variation in the data
    can only come from binomial sampling noise under those models.

    Higher T = more stimulus-level variability within same-side groups than models predict.
    Catches alternation_bias and griffiths variants (all dramatically under-predict within-group SD).
    """
    def alt_rate(seq):
        if len(seq) < 2:
            return 0.0
        return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1]) / (len(seq) - 1)

    group_left = []   # stimuli where alternating seq is on left
    group_right = []  # stimuli where alternating seq is on right

    for r in rows:
        p_left = float(r["chose_left_pct"])
        ar_a = alt_rate(r["sequence_a"])
        ar_b = alt_rate(r["sequence_b"])
        if ar_a > ar_b:
            group_left.append(p_left)
        elif ar_b > ar_a:
            group_right.append(p_left)

    def sd(values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    sds = []
    if group_left:
        sds.append(sd(group_left))
    if group_right:
        sds.append(sd(group_right))

    return sum(sds) / len(sds) if sds else 0.0
