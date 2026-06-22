import numpy as np

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    def kl_divergence_runs(seq: str) -> float:
        if not seq: return 0.0
        runs = []
        current_run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        if not runs: return 0.0
        
        # Count frequencies
        counts = {}
        for r in runs:
            counts[r] = counts.get(r, 0) + 1
            
        n_runs = len(runs)
        kl = 0.0
        for k, count in counts.items():
            obs_prob = count / n_runs
            # Expected probability for run length k in a finite sequence is tricky, 
            # but asymptotically it's 0.5^k.
            # To make it a proper distribution over observed keys, we shouldn't normalize.
            # Actually, standard KL divergence sum(P * log(P/Q))
            exp_prob = 0.5 ** k
            kl += obs_prob * np.log(obs_prob / exp_prob)
            
        return float(kl)

    return {
        "kl_runs_a": kl_divergence_runs(sequence_a),
        "kl_runs_b": kl_divergence_runs(sequence_b)
    }

print(compute_features("HTHTHTHT", "HHTTHHTT"))
print(compute_features("HHHHHHHH", "HTHTHTHT"))
