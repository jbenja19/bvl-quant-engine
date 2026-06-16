def transform_file(filepath, t1_idx, t1_len, t2_idx, t2_len):
    lines = open(filepath, 'r', encoding='utf-8').read().splitlines()
    
    # Transform Table 1
    lines[t1_idx] = '\\begin{tabular}{l l}'
    for i in range(t1_idx + 4, t1_idx + 4 + t1_len):
        parts = lines[i].split(' & ', 1)
        if len(parts) == 2:
            # isolate the content before any trailing \\ or \\[4pt]
            tail_idx = parts[1].find(r' \\')
            if tail_idx == -1: tail_idx = len(parts[1])
            content = parts[1][:tail_idx]
            tail = parts[1][tail_idx:]
            lines[i] = parts[0] + ' & \\parbox[t]{12.5cm}{' + content + '}' + tail

    # Transform Table 2
    lines[t2_idx] = '\\begin{tabular}{l l l}'
    for i in range(t2_idx + 4, t2_idx + 4 + t2_len):
        parts = lines[i].split(' & ', 2)
        if len(parts) == 3:
            tail_idx = parts[2].find(r' \\')
            if tail_idx == -1: tail_idx = len(parts[2])
            content = parts[2][:tail_idx]
            tail = parts[2][tail_idx:]
            lines[i] = parts[0] + ' & ' + parts[1] + ' & \\parbox[t]{8.5cm}{' + content + '}' + tail

    open(filepath, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

transform_file('Quant_Report_BVL_ES.tex', 132, 6, 420, 6)
transform_file('Quant_Report_BVL_EN.tex', 132, 6, 414, 6)
