import itertools as it
import multiprocessing as mp
from goethe.corpora import LeipzigCorpus
from gensim.models import word2vec


model_configurations = {
    'size': [50, 100, 300, 600],
    'iter': [3, 5, 10, 20, 50]
}

sample_sizes = [int(n) for n in [500e3, 1e6, 5e6, 10e6, 20e6]]


def train_model(sample_size, vec_size, epochs):
    sentences = LeipzigCorpus('data', max_sentences=sample_size)
    model = word2vec.Word2Vec(sentences=sentences, size=vec_size, iter=epochs)
    name = 'n{0}_size{1}_epochs{2}'.format(sample_size, vec_size, epochs)
    return name, model


if __name__ == '__main__':
    parameters = it.product(sample_sizes,
                            model_configurations['size'],
                            model_configurations['iter'])

    with mp.Pool() as pool:
        for name, model in pool.starmap(func=train_model,
                                        iterable=parameters,
                                        chunksize=1):
            model.save('models/' + name + '.model')
            print('Saved model: ' + name)
