import re
import os
import pandas as pd
import numpy as np
import json
from gensim.models.word2vec import Word2Vec

QUESTIONS = 'evaluation/question-words.txt'


def acc_summary(section, correct, incorrect):
    return {
        'name': section,
        'questions': len(correct) + len(incorrect),
        'accuracy': round(len(correct) / (len(correct) + len(incorrect)), 2)
    }


def split_modelname(name):
    name = name.replace('.model', '')
    params = re.findall(r'[a-z]+', name)
    values = [int(v) for v in re.findall(r'\d+', name)]
    return dict(zip(params, values))


def gridsearch_evaluation():
    models = [m for m in os.listdir('models') if m.endswith('.model')]
    paths = [os.path.join('models', m) for m in models]
    confs = [split_modelname(m) for m in models]
    for i, (path, conf) in enumerate(zip(paths, confs)):
        model = Word2Vec.load(path)
        sec_accuracies = model.accuracy(QUESTIONS)
        results = [{**acc_summary(**sec), **conf}
                   for sec in sec_accuracies]
        with open('evaluation/results.txt', 'a') as f:
            f.writelines([json.dumps(r, ensure_ascii=False) + '\n'
                          for r in results])

        print('({} of {}) '.format(i+1, len(paths)) +
              'Wrote results for n: {n}, size: {size}, epochs: {epochs}, sg: {sg}'.format(**conf))


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
