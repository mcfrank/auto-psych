def compute_features(sequence_a: str, sequence_b: str) -> dict:
    def transition_deviations(seq: str) -> float:
        if len(seq) < 2: return 0.0
        n_hh, n_ht, n_th, n_tt = 0, 0, 0, 0
        for i in range(len(seq)-1):
            if seq[i:i+2] == 'HH': n_hh += 1
            elif seq[i:i+2] == 'HT': n_ht += 1
            elif seq[i:i+2] == 'TH': n_th += 1
            elif seq[i:i+2] == 'TT': n_tt += 1
        
        p_h_given_h = n_hh / (n_hh + n_th) if (n_hh + n_th) > 0 else 0.5
        p_h_given_t = n_ht / (n_ht + n_tt) if (n_ht + n_tt) > 0 else 0.5
        
        # Mean squared deviation from 0.5
        return (p_h_given_h - 0.5)**2 + (p_h_given_t - 0.5)**2

    return {
        "trans_dev_a": transition_deviations(sequence_a),
        "trans_dev_b": transition_deviations(sequence_b)
    }

print(compute_features("HTHTHTHT", "HHHHHHHH"))
print(compute_features("HHTTHHTT", "HTHTHTHT"))
