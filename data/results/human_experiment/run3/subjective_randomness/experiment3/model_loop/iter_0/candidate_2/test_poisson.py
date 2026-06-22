import math
def compute_features(sequence_a: str, sequence_b: str) -> dict:
    def get_s_fact(seq):
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
        
        s_fact = sum(math.log(math.factorial(r - 1)) for r in runs)
        return float(s_fact)
        
    return {
        "s_fact_a": get_s_fact(sequence_a),
        "s_fact_b": get_s_fact(sequence_b)
    }

print(compute_features("HHTTHH", "HHHHTT"))
