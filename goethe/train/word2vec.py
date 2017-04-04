import argparse
from gensim.models.word2vec import Word2Vec, LineSentence

description = """
Script for training word vector models using public corpora
"""


def word2vec(train, output, limit=None, **w2vargs):
    if limit:
        sents = LineSentence(train, int(limit))
    else:
        sents = LineSentence(train)
    model = Word2Vec(sents, **w2vargs)
    model.wv.save_word2vec_format(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-t', '--train', type=str, required=True,
                        help='token file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Target file to store model')
    parser.add_argument('-limit', type=float,
                        help='Number of sentences to use from data')
    parser.add_argument('-workers', type=int, default=8,
                        help='number of worker threads to train the model')
    modelargs = parser.add_argument_group(title='word2vec')
    modelargs.add_argument('-iter', type=int, default=5,
                           help='Number of iterations over the corpus')
    modelargs.add_argument('-size', type=int, default=400,
                           help='Size of word vectors')
    modelargs.add_argument('-window', type=int, default=5,
                           help='Size of the sliding window')
    modelargs.add_argument('-min_count', type=int, default=5,
                           help='Ignore all words with lower total frequency')
    modelargs.add_argument('-sample', type=float, default=0.00001,
                           help='Threshold frequency over which words are downsampled')
    model = modelargs.add_mutually_exclusive_group()
    model.add_argument('-sg', dest='sg', action='store_const', const=1,
                       help='Use Skip-Gram model')
    model.add_argument('-cbow', dest='sg', action='store_const', const=0,
                       help='Use CBOW model')
    args = parser.parse_args()

    word2vec(**vars(args))
