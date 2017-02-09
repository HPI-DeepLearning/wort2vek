from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Activation
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
import numpy as np
import itertools
from itertools import chain
from collections import Counter
from .corpora import Corpus
import os
import random

class TrainingData(object):
    UNKNOWN_TOKEN = 'unk'
    TRAIN_PROB = 0.8

    def __init__(self, corpus, restrict_vocab=None, split=True):
        self.corpus = corpus
        self.nb_words = restrict_vocab
        self.count = None
        self.word2index = None
        self.index2word = None
        self.split = split
        self.build_indices()
        if self.split:
            self.maybe_split_corpus()

    def token_file(self, suffix=None):
        path, ext = os.path.splitext(self.corpus.file_path(use_tokens=True))
        return path + suffix + ext if suffix else path + ext

    @property
    def fn_train(self):
        return self.token_file('.train') if self.split else self.token_file()

    @property
    def fn_test(self):
        return self.token_file('.test') if self.split else self.token_file()

    @property
    def voc_size(self):
        return len(self.word2index)

    @classmethod
    def from_path(cls, path, restrict_vocab=None, split=True):
        corpus = Corpus(path)
        return cls(corpus, restrict_vocab=restrict_vocab, split=split)

    def maybe_split_corpus(self):
        if not os.path.isfile(self.fn_train):
            with open(self.fn_train, 'w') as f_train, \
                  open(self.fn_test, 'w') as f_test:
                for line in self.corpus.token_lines():
                    if random.uniform(0,1) < self.TRAIN_PROB:
                        f_train.write(line)
                    else:
                        f_test.write(line)

    def build_indices(self):
        self.count = [[self.UNKNOWN_TOKEN, -1]]

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

    def in_vocab(self, word):
        return word in self.word2index

    def sentence_to_numbers(self, sentence):
        numbers = list()
        for word in sentence:
            if self.in_vocab(word):
                index = self.word2index[word]
            else:
                index = 0  # dictionary[UNKNOWN_TOKEN]
            numbers.append(index)
        return numbers

    def numbers_to_sentence(self, numbers):
        return [self.index2word[index] for index in numbers]

    def tokens(self, is_train):
        path = self.fn_train if is_train else self.fn_test
        with open(path) as f:
            yield from (line.strip().split() for line in f)

    def dataset(self, is_train):
        """Return iterator of arrays of numbers representing words, accoring to
           word2index. The numbers can be transformed to words with index2word.
        """
        for sentence in self.tokens(is_train):
            yield self.sentence_to_numbers(sentence)

    def n_samples(self, is_train=True):
        path = self.fn_train if is_train else self.fn_test
        length = 0
        with open(path) as f:
            for _ in f:
                length += 1
        return length

    def batches(self, batch_size, is_train=True):
        """Return generator of training batches
        """
        while True:
            batch = []

            for item in self.dataset(is_train):
                if len(batch) == batch_size:
                    yield batch
                    batch = []
                batch.append(item)

            if len(batch):
                yield batch

class NgramTrainingData(TrainingData):
    def __init__(self, corpus, n, restrict_vocab=None):
        super().__init__(corpus, restrict_vocab)
        self.n = n

    @classmethod
    def from_path(cls, path, n, limit=None, restrict_vocab=None):
        corpus = Corpus(path, limit=limit)
        return cls(corpus, n, restrict_vocab=restrict_vocab)

    def ngrams(self, numbers):
        for i in range(0, len(numbers) - self.n + 1):
            yield numbers[i:i+self.n]

    def dataset(self, is_train):
        for numbers in super().dataset(is_train):
            yield from self.ngrams(numbers)


class SentencePartTrainingData(TrainingData):
    def dataset(self, is_train):
        for sentence in super().dataset(is_train):
            for i in range(0,len(sentence)):
                yield sentence[:i], sentence[i]


class NNLM(object):
    def __init__(self, train_data, word2vec):
        self.train_data = train_data
        self.word2vec = word2vec

    @property
    def voc_size(self):
        return self.train_data.voc_size

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

    def embedding_layer(self, n_inputs, mask_zero=True):
        return Embedding(self.voc_size + 1,
                         self.vector_size,
                         weights=[self.embedding_matrix()],
                         input_length=n_inputs,
                         mask_zero=mask_zero,
                         trainable=False)

    def batches(self, batch_size, is_train=True):
        for batch in self.train_data.batches(batch_size, is_train):
            yield self.split_batch(batch)

    def train(self, epochs=1, batch_size=32, model=None):
        if not model:
            model = self.model()
            self.compile_model(model)
        model.fit_generator(self.batches(batch_size),
                            samples_per_epoch=self.train_data.n_samples(),
                            nb_epoch=epochs,
                            verbose=1)
        return model

    def test(self, model, batch_size=32):
        return model.evaluate_generator(
            self.batches(batch_size, is_train=False),
            self.train_data.n_samples(is_train=False))

    def prob(self, model, sentence):
        predictions = self.predict(model, sentence[:-1])
        offset = len(sentence) - len(predictions)
        target = self.train_data.sentence_to_numbers(sentence)[offset:]
        prob = 1
        for i, prediction in enumerate(predictions):
            prob *= prediction[target[i]]
        return prob

    def predict_words(self, model, sentence):
        predictions = self.predict(model, sentence)
        nums = np.argmax(predictions, axis=1)
        return self.train_data.numbers_to_sentence(nums)

    def topn_next_words(self, model, sentence, n=3):
        res = self.predict(model, sentence)[0][-1]
        index = self.train_data.index2word
        pairs = [(index[i], p) for i, p in enumerate(res)]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:n]

    def predict(self, model, sentence):
        raise NotImplementedError

    def model(self):
        raise NotImplementedError

    def compile_model(self, model):
        raise NotImplementedError

    def split_batch(self, data):
        raise NotImplementedError


class NgramNNLM(NNLM):

    HIDDEN_PER_INPUT = 400

    def __init__(self, train_data, word2vec):
        super().__init__(train_data, word2vec)
        self.n_hidden = self.HIDDEN_PER_INPUT * (self.train_data.n - 1)

    @staticmethod
    def split_batch(triples):
        X = []
        y = []
        for triple in triples:
            X += [triple[:-1]]
            y += [triple[-1]]
        return np.array(X), np.array(y)

    def model(self):
        model = Sequential()
        model.add(self.embedding_layer(self.train_data.n - 1, mask_zero=False))
        model.add(Flatten())
        model.add(Dense(self.n_hidden, activation='relu'))
        model.add(Dense(self.voc_size, activation='softmax'))
        return model

    @staticmethod
    def compile_model(model):
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

    def predict(self, model, sentence):
        numbers = self.train_data.sentence_to_numbers(sentence)
        ngrams = self.train_data.ngrams(numbers)
        ngram_input = [ngram[:-1] for ngram in ngrams]
        predictions = model.predict(np.asarray(ngram_input))
        return predictions

class RnnNNLM(NNLM):
    def __init__(self, train_data, word2vec, max_sequence_length=50):
        super().__init__(train_data, word2vec)
        self.n_inputs = max_sequence_length
        self.n_hidden = 500

    def model(self):
        model = Sequential()
        model.add(self.embedding_layer(self.n_inputs))
        model.add(LSTM(self.n_hidden,
                       input_length=self.n_inputs,
                       return_sequences=True,
                       consume_less='mem'))
        model.add(TimeDistributed(Dense(output_dim=self.voc_size,
                                        input_dim=self.n_hidden)))
        model.add(Activation('softmax'))
        return model

    def predict(self, model, sentence):
        numbers = self.train_data.sentence_to_numbers(sentence)
        padded = pad_sequences([numbers], self.n_inputs, padding='post')
        predictions = model.predict(np.asarray(padded))[0]
        return predictions

    @staticmethod
    def compile_model(model):
        model.compile(optimizer='rmsprob',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

    def split_batch(self, sentences):
        X = list()
        y = list()
        for sentence in sentences:
            X.append(sentence[:-1])
            y.append(sentence[1:])
        X = pad_sequences(X, maxlen=self.n_inputs, padding='post')
        y = pad_sequences(y, maxlen=self.n_inputs, padding='post')
        X = np.asarray(X, dtype='int32')
        y = np.asarray(y, dtype='int32')
        y = np.expand_dims(y, -1)
        return X, y
