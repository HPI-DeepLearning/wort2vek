from goethe.nnlm import TrainingData, RnnNNLM
from gensim.models.word2vec import Word2Vec
from optparse import OptionParser

def train_lstm_nnlm(data_path, vector_file, epochs=1, batch_size=32):
    training_data = TrainingData.from_path(data_path)
    word2vec = Word2Vec.load(vector_file)
    word2vec.init_sims(replace=True)

    rnn_nnlm = RnnNNLM(training_data, word2vec)
    model = rnn_nnlm.train(int(epochs), int(batch_size))
    return model

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="data_path", metavar="PATH")
    parser.add_option("-v", "--vectors", dest="vector_file", metavar="FILE")
    parser.add_option("-e", "--epochs", dest="epochs", metavar="NUMBER")
    parser.add_option("-b", "--batch-size", dest="batch_size", metavar="NUMBER")
    (options, args) = parser.parse_args()
    param_dict = dict((k, v) for k, v in vars(options).items() if v)
    train_lstm_nnlm(**param_dict)
