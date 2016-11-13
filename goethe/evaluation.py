def best_match(model, questions_file, topn=3):
    for line in open(questions_file):
        if line.startswith(':'):
            continue

        king, queen, man, woman = line.strip().split()
        print('\n{king} is to {man}, what ({queen}) is to {woman}:'.format(
            man=man,
            king=king,
            woman=woman,
            queen=queen
        ))
        matches = model.most_similar(positive=[king, woman],
                                     negative=[man],
                                     topn=topn)
        for match in matches:
            cosine_similarity = match[1]
            term = match[0]
            print('\t{:.2}\t{}'.format(cosine_similarity, term))


def load_questions(filename):
    """Load file and store pairs grouped by sections.
    """
    sections = []
    for line in open(filename):
        if line.startswith(':'):
            sections.append({
                'section': line.strip(),
                'questions': []
            })
            continue
        sections[-1]['questions'].append(line.strip().split())
    return sections


def model_accuracy(model, questions_file, topn=1):
    section_evaluations = []
    for section in load_questions(questions_file):
        hits = []
        for king, queen, man, woman in section['questions']:
            try:
                matches = [term for term, acc in
                           model.most_similar(positive=[king, woman],
                                              negative=[man], topn=topn)]
            except KeyError:
                # Thrown when a word is not contained in the model's vocabulary
                matches = []
            hits.append(queen in matches)
        section_evaluations.append((section['section'].split()[1],
                                    sum(hits) / len(hits)))
    return section_evaluations
