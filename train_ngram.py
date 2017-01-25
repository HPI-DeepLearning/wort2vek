from goethe.corpora import LineNumbers
from goethe.nnlm import TrainingData, NgramNNLM

corpus = LineNumbers('preprocessed/large/numbers.txt')
training_data = TrainingData(corpus)
training_data.build_indices()

word2vec = Word2Vec.load('models/preprocessed-tokens-300-cb.model')
word2vec.init_sims(replace=True)

nnlm = NgramNNLM(word2index=training_data.word2index, word2vec=word2vec)

model = nnlm.train(training_data.ngram_dataset(3), model, nb_epoch=10, samples_per_epoch=1000000)
