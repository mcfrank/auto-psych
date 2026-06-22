def get_counts(seq):
    counts = {'rep': [0]*8, 'alt': [0]*8}
    if not seq: return counts
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            counts['rep'][current_run] += 1
            current_run += 1
        else:
            counts['alt'][current_run] += 1
            current_run = 1
    return counts

print(get_counts("HHHHTT"))
print(get_counts("HHTTHH"))
