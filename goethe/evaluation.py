from gensim.models.word2vec import Word2Vec
import os
import pandas as pd
import numpy as np

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

def summarize_result(result):
    for section in result:
        n_correct = len(section['correct'])
        n_incorrect = len(section['incorrect'])
        yield {
            'name': section['section'],
            'acc': n_correct / (n_correct + n_incorrect),
            'correct': n_correct,
            'incorrect': n_incorrect
        }

def gensim_model_acc(model, questions_file):
    results = model.accuracy(questions_file)
    return summarize_result(results)

def split_modelname(name):
    parts = name.split('.')[0].split('_')
    n_sentences = int(parts[0].replace('n', ''))
    vector_size = int(parts[1].replace('size', ''))
    epochs = int(parts[2].replace('epochs', ''))
    return n_sentences, vector_size, epochs

def eval_models(model_files, params_list, questions_file):
    for (model_file, params) in zip(model_files, params_list):
        model = Word2Vec.load(model_file)
        res = gensim_model_acc(model, questions_file)
        n_sentences, vector_size, epochs = params
        _, model_name = os.path.split(model_file)
        yield {
            'name': model_name,
            'params': {
                'vector_size': vector_size,
                'epochs': epochs,
                'n_sentences': n_sentences
            },
            'eval_results': list(res)
        }

def eval_models_from_gridsearch(model_names, model_path, questions_file):
    params = [split_modelname(name) for name in model_names]
    model_files = [os.path.join(model_path, name) for name in model_names]
    return eval_models(model_files, params, questions_file)

def eval_dict_to_dataframe(eval_dicts):
    eval_dicts = list(eval_dicts)
    columns = [d['name'] for d in eval_dicts[0]['eval_results']]
    column_names = ['name', 'n_sentences', 'vector_size', 'epochs'] + columns
    rows = []
    for eval_dict in eval_dicts:
            vector_size, epochs, n_sentences = list(eval_dict['params'].values())
            accs = [d['acc'] for d in eval_dict['eval_results']]
            row = [eval_dict['name'], n_sentences, vector_size, epochs]
            row += list(accs)
            rows.append(row)
    return pd.DataFrame(np.array(rows), columns=column_names)
