from goethe.corpora import LineNumbers
from goethe.nnlm import TrainingData, RnnNNLM
from gensim.models.word2vec import Word2Vec


def lstm_nnlm():
    corpus = LineNumbers('preprocessed/large/numbers.txt')
    training_data = TrainingData(corpus)
    training_data.build_indices()

    word2vec = Word2Vec.load('models/preprocessed-tokens-300-cb.model')
    word2vec.init_sims(replace=True)

    rnn_nnlm = RnnNNLM(word2index=training_data.word2index,
                       word2vec=word2vec, max_sequence_length=30)
    model = rnn_nnlm.train(training_data.sentence_dataset(),
                           nb_epoch=10, samples_per_epoch=1000000)
    return model


if __name__ == "__main__":
    lstm_nnlm()
