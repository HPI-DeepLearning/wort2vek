import argparser
from corpora import Corpus


def word2vec(corpus, target, **w2vargs):
    corpus = Corpus(corpus)
    model = gensim.models.Word2Vec(corpus, **w2vargs)
    model.save_word2vec_format(target, binary=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for training word vector models using public corpora')
    parser.add_argument('corpus', type=str,
                        help='source folder with preprocessed corpus (one sentence plain text per line in each file)')
    parser.add_argument('target', type=str,
                        help='target file name to store model in')
    parser.add_argument('-s', '--size', type=int, default=300,
                        help='dimension of word vectors')
    parser.add_argument('-w', '--window', type=int, default=5,
                        help='size of the sliding window')
    parser.add_argument('-m', '--mincount', type=int, default=5,
                        help='minimum number of occurences of a word to be considered')
    parser.add_argument('-c', '--workers', type=int, default=4,
                        help='number of worker threads to train the model')
    parser.add_argument('-g', '--sg', type=int, default=1,
                        help='training algorithm: Skip-Gram (1), otherwise CBOW (0)')
    parser.add_argument('-i', '--hs', type=int, default=1,
                        help='use of hierachical sampling for training')
    parser.add_argument('-n', '--negative', type=int, default=0,
                        help='use of negative sampling for training (usually between 5-20)')
    parser.add_argument('-o', '--cbowmean', type=int, default=0,
                        help='for CBOW training algorithm: use sum (0) or mean (1) to merge context vectors')
    args = parser.parse_args()

    word2vec(**args)
