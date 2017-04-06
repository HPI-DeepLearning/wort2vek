import fasttext
import argparse
from ..utils import args_to_kwargs

"""
Reference: https://github.com/salestock/fastText.py
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fasttext training')
    parser.add_argument('--input_file', type=str,
                        help='training file path (required)')
    parser.add_argument('--output', type=str,
                        help='output file path (required)')
    modelargs = parser.add_argument_group('fasttext')
    modelargs.add_argument('--lr', type=float, help='learning rate [0.05]')
    modelargs.add_argument('--lr_update_rate', type=int,
                           help='change the rate of updates for the learning rate [100]')
    modelargs.add_argument('--dim', type=int, help='size of word vectors [100]')
    modelargs.add_argument('--ws', type=int,
                           help='size of the context window [5]')
    modelargs.add_argument('--epoch', type=int, help='number of epochs [5]')
    modelargs.add_argument('--min_count', type=int,
                           help='minimal number of word occurences [5]')
    modelargs.add_argument('--neg', type=int,
                           help='number of negatives sampled [5]')
    modelargs.add_argument('--word_ngrams', type=int,
                           help='max length of word ngram [1]')
    modelargs.add_argument('--loss', type=str,
                           choices=['ns', 'hs', 'softmax'], help='loss function {ns, hs, softmax} [ns]')
    modelargs.add_argument('--bucket', type=int,
                           help='number of buckets [2000000]')
    modelargs.add_argument('--minn', type=int,
                           help='min length of char ngram [3]')
    modelargs.add_argument('--maxn', type=int,
                           help='max length of char ngram [6]')
    modelargs.add_argument('--thread', type=int, help='number of threads [12]')
    modelargs.add_argument('--t', type=float, help='sampling threshold [0.0001]')
    modelargs.add_argument('--silent', type=int,
                           help='disable the log output from the C++ extension [1]')
    modelargs.add_argument('--encoding', type=str,
                           help='specify input_file encoding [utf-8]')
    model = modelargs.add_mutually_exclusive_group(required=True)
    model.add_argument('--sg', action='store_true', dest='sg',
                       help='use skip-gram')
    model.add_argument('--cbow', action='store_false', dest='sg',
                       help='use CBOW')
    args = parser.parse_args()
    kwargs = args_to_kwargs(args)
    del kwargs['sg']

    if args.sg:
        fasttext.skipgram(**kwargs)
    else:
        fasttext.cbow(**kwargs)
