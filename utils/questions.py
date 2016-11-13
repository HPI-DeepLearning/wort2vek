import itertools as it


def quadruples_from_pairs(in_filename, out_filename):
    # Load file and store pairs grouped by sections
    sections = []
    for line in open(in_filename):
        if line.startswith(':'):
            sections.append({
                'name': line.strip(),
                'pairs': []
            })
            continue
        sections[-1]['pairs'].append(line.strip().split('-'))

    # For each sections generate 4-tuples
    for section in sections:
        section['4-tuples'] = [list(it.chain(*c)) for c
                               in it.combinations(section['pairs'], 2)]
        del section['pairs']

    # Write 4-tupes to file
    lines = []
    for s in sections:
        lines.append(s['name'])
        lines.extend(' '.join(t) for t in s['4-tuples'])
    with open(out_filename, 'w') as f:
        f.writelines(l + '\n' for l in lines)
    return out_filename
