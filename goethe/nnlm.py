from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
import numpy as np
import itertools

class NNLM:

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

    def split_triples(self, triples):
        X = []
        y = []
        for triple in triples:
            X += [triple[:-1]]
            y += [triple[-1]]
        return np.array(X), np.array(y)

    def train_data_generator(self, triples, batch_size=32):
        while True:
            batch = itertools.islice(triples, batch_size)
            if not batch:
                break
            yield self.split_triples(batch)

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
        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

        train_gen = self.train_data_generator(train_data_triples)

        model.fit_generator(train_gen,
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=nb_epoch,
                            verbose=1)
        return model
