import time
import itertools as it
import multiprocessing as mp
from goethe.corpora import LeipzigCorpus
from gensim.models import word2vec

model_config = {
    'size': [100, 300, 600],
    'iter': [5, 10, 20],
    'sg': [0, 1]  # Skip-gram
}

sample_sizes = [int(n) for n in [1e6, 3e6, 10e6, 20e6]]


def train_model(config):
    sample, size, epochs, sg = config
    sentences = LeipzigCorpus('data', max_sentences=sample)
    model = word2vec.Word2Vec(sentences=sentences, size=size, iter=epochs,
                              workers=4)
    name = 'n{}_size{}_epochs{}_sg{}'.format(sample, size, epochs, sg)
    return name, model


def minutes(t0):
    t1 = time.time()
    return int((t1-t0)/60)


if __name__ == '__main__':
    parameters = it.product(sample_sizes, model_config['size'],
                            model_config['iter'], model_config['sg'])
    t0 = time.time()
    for p in parameters:
        name, model = train_model(p)
        model.save('models/' + name + '.model')
        print('{}\', saved model: {}'.format(minutes(t0), name))


