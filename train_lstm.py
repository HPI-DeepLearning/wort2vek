from argparse import ArgumentParser

def train_lstm_nnlm(data_path, vector_file, epochs=1, batch_size=32,
                    restrict_vocab=100000, model_file=None, max_sequence_length=40):
    from goethe.nnlm import TrainingData, RnnNNLM
    from gensim.models.word2vec import Word2Vec
    # Fail early if h5py is not installed
    if model_file:
        import h5py

    training_data = TrainingData.from_path(data_path, restrict_vocab=restrict_vocab)
    word2vec = Word2Vec.load(vector_file)
    word2vec.init_sims(replace=True)

    rnn_nnlm = RnnNNLM(training_data, word2vec, max_sequence_length)
    model = rnn_nnlm.train(epochs, batch_size)
    rnn_nnlm.test(model)
    if model_file:
        model.save(model_file)
    return model

if __name__ == "__main__":
    parser = ArgumentParser(description='Train a Neural Language model with a LSTM.')

    parser.add_argument("-p", "--path", dest="data_path", metavar="DATA_PATH",
        required=True, help="Path to directory containing training data.")
    parser.add_argument("-v", "--vectors", dest="vector_file", required=True,
        metavar="VECTORS_FILE", help="File with word vectors, built with gensim.")
    parser.add_argument("-s", "--save_file", dest="model_file",
        metavar="SAVE_MODEL_FILE", help="Filename to save the model.")
    parser.add_argument("-e", "--epochs", dest="epochs", metavar="EPOCHS",
        type=int, help="Number of epochs to run over all training data.")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int,
        metavar="BATCH_SIZE", help="Number of sentences processed in one batch.")
    parser.add_argument("-r", "--restrict-vocab", dest="restrict_vocab", type=int,
        metavar="VOCABS_SIZE", help="Max vocab size. Can reduce memory consumption heavily")
    parser.add_argument("-m", "--max-length", dest="max_sequence_length", type=int,
        metavar="MAX_SENTENCE_LENGTH", help="Max length of a sentence for lstm.")

    args = parser.parse_args()
    param_dict = dict((k, v) for k, v in vars(args).items() if v)
    train_lstm_nnlm(**param_dict)
