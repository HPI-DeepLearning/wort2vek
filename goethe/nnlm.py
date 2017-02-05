from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Activation
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
import numpy as np
import itertools
from itertools import chain
from collections import Counter

class TrainingData(object):

    UNKNOWN_TOKEN = "unk"

    def __init__(self, corpus, restrict_vocab=None):
        self.corpus = corpus
        self.nb_words = restrict_vocab

        self.count = None
        self.word2index = None
        self.index2word = None
        self.build_indices()

    def __len__(self):
        return len(self.word2index)

    def build_indices(self):
        self.count = [[TrainingData.UNKNOWN_TOKEN, -1]]

        counter = Counter(chain.from_iterable(self.corpus))

        if self.nb_words:
            common_counts = counter.most_common(self.nb_words - 1)
        else:
            common_counts = counter.most_common()

        self.count.extend(common_counts)

        self.word2index = dict()
        for word, _ in self.count:
            self.word2index[word] = len(self.word2index)

        unk_count = 0
        for sentence in self.corpus:
            for word in sentence:
                if word not in self.word2index:
                    unk_count += 1
        self.count[0][1] = unk_count

        self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))

    def number_sentences(self):
        for sentence in self.corpus:
            numbers = list()
            for word in sentence:
                if word in self.word2index:
                    index = self.word2index[word]
                else:
                    index = 0  # dictionary[UNKNOWN_TOKEN]
                numbers.append(index)
            yield numbers

    def dataset(self):
        """Return generator of training data
        """
        raise NotImplementedError("Subclass responsibility")

    def batches(self, data, batch_size):
        """Return generator of training batches
        """
        data = self.dataset()
        while True:
            batch_data = itertools.islice(data, batch_size)
            if not batch_data:
                return
            yield batch_data


class NgramTrainingData(TrainingData):

    def __init__(self, corpus, n, restrict_vocab=None):
        super().__init__(corpus, restrict_vocab)
        self.n = n

    def dataset(self):
        for numbers in self.number_sentences():
            for i in range(0, len(numbers) - n + 1):
                yield numbers[i:i+n]


class SentenceTrainingData(TrainingData):

    def dataset(self):
        return self.number_sentences()


class SentencePartTrainingData(TrainingData):

    def dataset(self):
        for sentence in self.number_sentences():
            for i in range(0,len(sentence)):
                yield sentence[:i], sentence[i]


class NNLM(object):

    def __init__(self, train_data):
        self.train_data = train_data

    @property
    def voc_size(self):
        return len(self.train_data)

    @property
    def vector_size(self):
        return self.word2vec.vector_size

    def embedding_matrix(self):
        embedding = np.zeros((self.voc_size + 1, self.vector_size))

        for word, i in self.train_data.word2index.items():
            try:
                embedding[i] = self.word2vec[word]
            except KeyError:
                pass

        return embedding

    def embedding_layer(self, n_inputs):
        return Embedding(self.voc_size + 1,
                         self.vector_size,
                         weights=[self.embedding_matrix()],
                         input_length=n_inputs,
                         trainable=False)

    def batches(self, batch_size=32):
        for batch in self.training_data.batches(batch_size):
            yield self.split_batch(batch)

    def train(self, batch_size=32, epochs=1):
        model = self.model()
        self.compile_model(model)
        for i in range(0, epochs):
            print("Running Epoch: %d" % i)
            model.fit_generator(self.batches(batch_size), verbose=1)
        return model

    def model(self):
        raise NotImplementedError("Subclass responsibility")

    def compile_model(self, model):
        raise NotImplementedError("Subclass responsibility")

    def split_batch(self):
        raise NotImplementedError("Subclass responsibility")


class NgramNNLM(NNLM):

    def __init__(self, train_data):
        super().__init__(train_data)
        self.n_hidden = 500

    def split_batch(self, triples):
        X = []
        y = []
        for triple in triples:
            X += [triple[:-1]]
            y += [triple[-1]]
        return np.array(X), np.array(y)

    def model(self):
        model = Sequential()
        model.add(self.embedding_layer(self.train_data.n))
        model.add(Flatten())
        model.add(Dense(self.n_hidden, activation='relu'))
        model.add(Dense(self.voc_size, activation='softmax'))
        return model

    def compile_model(self, model):
        model.compile(optimizer='sgd',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])


class RnnNNLM(NNLM):
    def __init__(self, train_data, max_sequence_length=40):
        super().__init__(train_data)
        self.n_inputs = max_sequence_length
        self.n_hidden = 300

    def model(self):
        model = Sequential()
        # TODO set mask_zero=True, ensure that there are no 0s in the vocab before
        model.add(self.embedding_layer(n_inputs))
        model.add(LSTM(self.n_hidden, input_length=self.n_inputs, return_sequences=True))
        model.add(TimeDistributed(Dense(output_dim=self.voc_size, input_dim=self.n_hidden)))
        model.add(Activation('softmax'))
        return model

    def compile_model(self, model):
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

    def split_data(self, sentences):
        X = list()
        y = list()
        for sentence in sentences:
            X.append(sentence)
            y.append(sentence[1:])
        X = np.asarray(pad_sequences(X, maxlen=self.n_inputs), dtype='int32')
        y = np.asarray(pad_sequences(X, maxlen=self.n_inputs), dtype='int32')
        y = np.expand_dims(y, -1)
        return X, y


class SentencePartRnnNNLM(NNLM):
    def __init__(self, train_data, max_sequence_length=40):
        super().__init__(train_data)
        self.n_inputs = max_sequence_length
        self.model = self.model()
        self.n_hidden = 200

    def model(self):
        model = Sequential()
        model.add(self.embedding_layer(n_inputs))
        model.add(LSTM(self.n_hidden))
        model.add(Dense(self.voc_size, activation='softmax'))
        return model

    def compile_model(self, model):
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

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
        X_pad = pad_sequences(X, maxlen=self.n_inputs)
        return np.asarray(X_pad, dtype='int32'), np.asarray(y, dtype='int32')
