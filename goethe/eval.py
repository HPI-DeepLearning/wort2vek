import re
import os
import random
import itertools as it
from collections import defaultdict

import gensim
import pandas as pd

import goethe


def summarize(sections):
    """Take sections as returned by `model.accuracy()` and return
    dict mapping section names to accuracies.
    """
    return {s['section']: round(len(s['correct'])
                                / (len(s['correct']) + len(s['incorrect'])), 2)
            for s in sections if len(s['correct']) + len(s['incorrect'])}


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


def gridsearch(data_path, gs_confs, sent_limits=None):
    """For each model/limit configuration yield a trained model and a model name.
    Takes dictionary of `configurations` for gensim's Word2Vec, e.g.:
        { 'size': [300, 600], 'sg': [0, 1] }
    and optional list of sentence limits.
    """
    confs = it.product(*gs_confs.values(),
                       sent_limits if sent_limits else [False])

    for *mconf, slimit in confs:
        mconf = {k: v for k, v in zip(gs_confs.keys(), mconf)}
        sents = goethe.corpora.Corpus(data_path, limit=slimit)
        model = gensim.models.Word2Vec(sentences=sents, **mconf, workers=8)
        yield model, model_name(mconf, slimit)


def model_name(conf, sentence_limit=None):
    """Join a model configuration and sentence limit into a model identifier.
    """
    tokens = ['%s%s' % (k, v) for k, v in conf.items()]
    if sentence_limit:
        tokens.append('n%s' % sentence_limit)
    return '_'.join(sorted(tokens))
