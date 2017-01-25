from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import itertools
from itertools import chain
from collections import Counter

class TrainingData(object):

    UNKNOWN_TOKEN = "unk"

    def __init__(self, corpus, nb_words=None):
        self.corpus = corpus
        self.nb_words = nb_words

        self.count = None
        self.word2index = None
        self.index2word = None

    def build_indices(self):
        self.count = [[TrainingData.UNKNOWN_TOKEN, -1]]

        counter = Counter(chain.from_iterable(self.corpus.sentences(words=True)))

        if self.nb_words:
            common_counts = counter.most_common(self.nb_words - 1)
        else:
            common_counts = counter.most_common()

        self.count.extend(common_counts)

        self.word2index = dict()
        for word, _ in self.count:
            self.word2index[word] = len(self.word2index)

        unk_count = 0
        for sentence in self.corpus.sentences(words=True):
            for word in sentence:
                if word not in self.word2index:
                    unk_count = unk_count + 1
        self.count[0][1] = unk_count

        self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))

    def sentence_dataset(self):
        for sentence in self.corpus.sentences(words=True):
            numbers = list()
            for word in sentence:
                if word in self.word2index:
                    index = self.word2index[word]
                else:
                    index = 0  # dictionary[UNKNOWN_TOKEN]
                numbers.append(index)
            yield numbers

    def ngram_dataset(self, n):
        for numbers in self.sentence_dataset():
            for i in range(0, len(numbers) - n + 1):
                yield numbers[i:i+n]


class NNLM(object):

    def __init__(self, word2index, word2vec):
        self.word2index = word2index
        self.word2vec = word2vec

    def nb_words(self):
        return len(self.word2index)

    def vector_size(self):
        return self.word2vec.vector_size

    def embedding_matrix(self):
        embedding = np.zeros((self.nb_words() + 1, self.vector_size()))

        for word, i in self.word2index.items():
            try:
                embedding[i] = self.word2vec[word]
            except KeyError:
                pass

        return embedding

    def embedding_layer(self, n_inputs):
        return Embedding(self.nb_words() + 1,
                         self.vector_size(),
                         weights=[self.embedding_matrix()],
                         input_length=n_inputs,
                         trainable=True)

    def train_data_generator(self, data, batch_size=32):
        while True:
            batch_data = itertools.islice(data, batch_size)
            if not batch_data:
                break
            yield self.split_data(batch_data)

    def split_data(self):
        raise Error("Subclass responsibility")

class NgramNNLM(NNLM):
    def __init__(self, word2index, word2vec):
        super().__init__(word2index, word2vec)

    def split_data(self, triples):
        X = []
        y = []
        for triple in triples:
            X += [triple[:-1]]
            y += [triple[-1]]
        return np.array(X), np.array(y)

    def model(self, n_inputs, n_hidden):
        model = Sequential()
        model.add(self.embedding_layer(n_inputs))
        model.add(Flatten())
        model.add(Dense(n_hidden, activation='relu'))
        model.add(Dense(self.nb_words(), activation='softmax'))
        return model

    def train(self, train_data_triples, model=None, samples_per_epoch=100000, nb_epoch=5):
        if not model:
            model = self.model(2, 500)
        model.compile(optimizer='sgd',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

        train_gen = self.train_data_generator(train_data_triples)

        model.fit_generator(train_gen,
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=nb_epoch,
                            verbose=1)
        return model

class RnnNNLM(NNLM):
    def __init__(self, word2index, word2vec, max_sequence_length=40):
        super().__init__(word2index, word2vec)
        self.n_inputs = max_sequence_length

    def model(self, n_inputs, n_hidden):
        model = Sequential()
        model.add(self.embedding_layer(n_inputs))
        model.add(LSTM(n_hidden))
        model.add(Dense(self.nb_words(), activation='softmax'))
        return model

    def sentence_parts(self, sentences):
        for sentence in sentences:
            for i in range(0,len(sentence)):
                yield sentence[:i], sentence[i]

    def split_data(self, parts):
        X = list()
        y = list()
        for part in parts:
            X.append(part[0])
            y.append(part[1])

        return pad_sequences(X, maxlen=self.n_inputs), np.array(y)

    def train(self, train_data, model=None, n_hidden=500, samples_per_epoch=10000, nb_epoch=1):
        model = self.model(self.n_inputs, n_hidden)
        model.compile(optimizer='sgd',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

        train_gen = self.train_data_generator(self.sentence_parts(train_data))

        model.fit_generator(train_gen,
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=nb_epoch,
                            verbose=1)
        return model
