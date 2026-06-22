def compute_features(seq_a, seq_b):
    def get_counts(s):
        s = s.upper().strip()
        counts = {'HH': 0, 'HT': 0, 'TH': 0, 'TT': 0}
        for i in range(len(s)-1):
            counts[s[i:i+2]] += 1
        return counts
    
    ca = get_counts(seq_a)
    cb = get_counts(seq_b)
    
    return {
        'n_HH_a': ca['HH'], 'n_HT_a': ca['HT'], 'n_TH_a': ca['TH'], 'n_TT_a': ca['TT'],
        'n_HH_b': cb['HH'], 'n_HT_b': cb['HT'], 'n_TH_b': cb['TH'], 'n_TT_b': cb['TT']
    }

print(compute_features('HHHH', 'HTHT'))
