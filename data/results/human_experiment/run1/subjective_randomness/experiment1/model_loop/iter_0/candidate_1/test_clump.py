def compute_features(sequence_a: str, sequence_b: str) -> dict:
    def clumpiness(seq):
        seq = seq.strip().upper()
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
        
        clump_items = sum(r for r in runs if r in (2, 3))
        return float(clump_items / len(seq))
        
    return {
        "clumpiness_a": clumpiness(sequence_a),
        "clumpiness_b": clumpiness(sequence_b)
    }

print(compute_features("HHTTHH", "HTHTHT"))
print(compute_features("HHHTTT", "HHHHHH"))
