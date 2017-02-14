import re
import os
import gensim
import random
import pandas as pd
from collections import defaultdict


def summarize(sections):
    """Take sections as returned by `model.accuracy()` and return
    dict mapping section names to accuracies.
    """
    return {s['section']: round(len(s['correct'])
                                / (len(s['correct']) + len(s['incorrect'])), 2)
            for s in sections}


def split_conf(name):
    """Split name by properties and values.
    Example:
        n10000000_size600_epochs10_sg0_window20.model
        {'epochs': 0, 'models': 10000000, 'n': 600, 'sg': 20, 'size': 10}
    """
    name = os.path.basename(name).replace('.model', '')
    params = re.findall(r'[a-z]+', name)
    values = [int(v) for v in re.findall(r'\d+', name)]
    return dict(zip(params, values))


def list_models(folder):
    """Return list of models found in `folder`cr
    """
    return [os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith('.model')]


def evaluate_model(model_path, questions_path):
    """Return accuracy values for model found at `model_path`
    with questions found at `questions_path`.
    """
    model = gensim.models.word2vec.Word2Vec.load(model_path)
    section_accs = model.accuracy(questions_path)
    return summarize(section_accs)


def gridsearch_evaluation(models_path, questions_path):
    """Evaluate models found in `models_path` and return a DataFrame of form:
        conf1 | ... | confK | section1 | section2 | ... | sectionN
        ----------------------------------------------------------
        valu1 | ... | valuK | accurac1 | accurac2 | ... | accuracN
    """
    paths = list_models(models_path)
    per_model_results = []
    for i, mpath in enumerate(paths):
        section_accs = evaluate_model(mpath, questions_path)
        conf = split_conf(mpath)
        per_model_results.append({**conf, **section_accs})
        print('({} of {}) '.format(i + 1, len(paths))
              + 'Wrote results for '
              + ', '.join(['{}: {}'.format(k, v) for k, v in conf.items()]))
    df = pd.DataFrame.from_records(per_model_results)
    df = df[list(conf) + list(section_accs)]
    return df


def question_counts(questions_path):
    """Return mapping from question categories to number of questions.
    """
    categories = defaultdict(int)
    with open(questions_path) as f:
        for line in f:
            if line.startswith(':'):
                category = line[2:].strip()
            else:
                categories[category] += 1
    return dict(categories)


def question_examples(questions_path, n=5):
    """Return mapping from question categories to `n` example questions.
    """
    ctgrs = defaultdict(list)
    with open(questions_path) as f:
        for i, line in enumerate(f):
            if line.startswith(':'):
                if i:
                    ctgrs[c] = random.sample(ctgrs[c], min(n, len(ctgrs[c])))
                c = line[2:].strip()
            else:
                ctgrs[c].append(line.strip())
    return dict(ctgrs)
