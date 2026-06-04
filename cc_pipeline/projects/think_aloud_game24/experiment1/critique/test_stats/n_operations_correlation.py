def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int),
          and optionally lm_code_translation_list (list of str).
    Returns Pearson r between model-predicted difficulty (1 - chose_left_pct) and mean
    number of explore_operation() calls per trace. Tests whether model difficulty tracks
    actual search depth. Returns 0.0 when no code traces are available.
    """
    import math

    difficulties = []
    mean_ops_list = []
    for row in rows:
        traces = row.get("lm_code_translation_list", [])
        if not traces:
            continue
        n_ops = sum(code.count("explore_operation(") for code in traces)
        mean_ops = n_ops / len(traces)
        difficulties.append(1.0 - row["chose_left_pct"])
        mean_ops_list.append(mean_ops)

    if len(difficulties) < 2:
        return 0.0

    n = len(difficulties)
    mean_d = sum(difficulties) / n
    mean_o = sum(mean_ops_list) / n
    cov = sum((difficulties[i] - mean_d) * (mean_ops_list[i] - mean_o) for i in range(n))
    sd_d = math.sqrt(sum((x - mean_d) ** 2 for x in difficulties))
    sd_o = math.sqrt(sum((x - mean_o) ** 2 for x in mean_ops_list))
    if sd_d == 0 or sd_o == 0:
        return 0.0
    return cov / (sd_d * sd_o)
