import itertools as it
from collections import defaultdict
import random

def tuples_to_sections(in_filename):
    # Load file and store pairs grouped by sections
    super_sections = []
    for line in open(in_filename):
        if line.startswith(':'):
            super_sections.append({
                'name': line.strip(),
                'tuples': []
            })
            continue
        super_sections[-1]['tuples'].append(line.strip().split(','))

    sections = []
    for super_section in super_sections:
        (name, subnames) = super_section['name'].split('-')
        subnames = subnames.strip().split(',')
        count = len(subnames)
        for i in range(1, count):
            section_name = name.strip() + ' - ' + subnames[0] + ', ' + subnames[i]
            pairs = [(t[0],t[i]) for t in super_section['tuples'] if len(t) > i]
            sections.append({ 'name': section_name, 'pairs': list(pairs) })

    return sections

def pairs_to_sections(in_filename):
    # Load file and store pairs grouped by sections
    sections = []
    for line in open(in_filename):
        if line.startswith(':'):
            sections.append({
                'name': line.strip(),
                'pairs': []
            })
            continue
        sections[-1]['pairs'].append(line.strip().split(','))
    return sections

def all_pairs(l):
    for e1 in l:
        for e2 in l:
            if e1 != e2:
                yield (e1, e2)

def sections_to_pairs(sections):
    # For each sections generate 4-tuples
    for section in sections:
        section['4-tuples'] = [list(it.chain(*c)) for c
                               in all_pairs(section['pairs'])]
        del section['pairs']

    # Create 4-tuples
    for s in sections:
        yield s['name']
        for t in s['4-tuples']:
            yield ' '.join(t)

def quadruples(filename):
    if filename.endswith('.pairs.txt'):
        sections = pairs_to_sections(filename)
    elif filename.endswith('.tuples.txt'):
        sections = tuples_to_sections(filename)
    else:
        raise ValueError("Unsupported file: " + filename)
    return sections_to_pairs(sections)

def files_to_quadtruples(in_filenames, out_filename):
    lines = it.chain(*[quadruples(f) for f in in_filenames])

    with open(out_filename, 'w') as f:
        f.writelines(l + '\n' for l in lines)
    return out_filename

def filter(questions, n=500):
    # Build sections
    sections = defaultdict(list)
    with open(questions) as f:
        for l in f.readlines():
            if ':' in l:
                sname = l
            else:
                sections[sname].append(l)

    # Write out
    with open(questions, 'w') as f:
        for sname, lines in sections.items():
            f.write(sname)
            # Sample
            if len(lines) >= n:
                lines = random.sample(lines, n)
            f.writelines(lines)


def count(questions):
    sections = defaultdict(int)
    with open(questions) as f:
        for l in f.readlines():
            if ':' in l:
                sname = l
            else:
                sections[sname] += 1
    return sections



in_path = 'evaluation/base/'
out_path = 'evaluation/'

def create(in_files, out_file):
    file_paths = [in_path + f for f in in_files]
    out_file = out_path + out_file
    files_to_quadtruples(file_paths, out_file)

def create_all():
    files =[
        'semantic.pairs.txt',
        'gender.pairs.txt',
        'grammatical-number.pairs.txt',
        'opposite.pairs.txt',
        'adjectives.tuples.txt',
        'verbs.tuples.txt'
    ]
    create(files, 'question-words.txt')

if __name__ == '__main__':
    create_all()
