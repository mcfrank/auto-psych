def compute_features(seq_a, seq_b):
    import re
    def run_lengths(seq):
        seq = seq.strip().upper()
        if not seq: return []
        runs = [len(m.group(0)) for m in re.finditer(r'(H+|T+)', seq)]
        return runs
    runs_a = run_lengths(seq_a)
    runs_b = run_lengths(seq_b)
    print("A:", runs_a, "B:", runs_b)
    
compute_features("HHTTHH", "HTHTHT")
