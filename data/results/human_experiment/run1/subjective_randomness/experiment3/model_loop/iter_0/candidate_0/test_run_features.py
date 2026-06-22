import re
import numpy as np
import pandas as pd

def compute_features(seq_a, seq_b):
    def get_runs(seq):
        seq = seq.strip().upper()
        if not seq: return []
        return [len(m.group(0)) for m in re.finditer(r'(H+|T+)', seq)]
    
    def sum_power(runs, power):
        return sum(r**power for r in runs)
        
    runs_a = get_runs(seq_a)
    runs_b = get_runs(seq_b)
    
    return {
        "run_sq_a": sum_power(runs_a, 2),
        "run_sq_b": sum_power(runs_b, 2),
        "run_cube_a": sum_power(runs_a, 3),
        "run_cube_b": sum_power(runs_b, 3)
    }

print(compute_features("HHTTHH", "HTHTHT"))
