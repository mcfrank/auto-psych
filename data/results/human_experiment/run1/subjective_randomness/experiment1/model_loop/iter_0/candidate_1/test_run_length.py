def max_run_of_char(seq):
    seq = seq.strip().upper()
    if not seq: return []
    runs = []
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    return runs

print(max_run_of_char("HHTTHH"))
print(max_run_of_char("HHHTTT"))
print(max_run_of_char("HTHTHT"))

