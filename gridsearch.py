import time
import itertools as it
from gensim.models import word2vec
from goethe.corpora import Corpus

model_config = {
    'size': [200, 300, 400, 500, 600],
    'window': [5, 10, 20],
    'sg': [0, 1]  # Skip-gram or CBOW
}

sample_size = 10000000
epochs = 10

def train_model(config):
    size, window, sg = config
    sentences = Corpus('../corpora/eval/eval.tokens.txt', limit=sample_size)
    model = word2vec.Word2Vec(sentences=sentences, size=size, window=window,
                              iter=epochs, workers=4)
    name = 'n{}_size{}_epochs{}_sg{}_window{}'.format(sample_size, size, epochs, sg, window)
    return name, model


def minutes(t0):
    t1 = time.time()
    return int((t1-t0)/60)


if __name__ == '__main__':
    parameters = it.product(model_config['size'], model_config['window']
                            model_config['sg'])
    t0 = time.time()
    for p in parameters:
        name, model = train_model(p)
        model.save('models/' + name + '.model')
        print('{}\', saved model: {}'.format(minutes(t0), name))
