import fasttext
import argparse

"""
Reference: https://github.com/salestock/fastText.py
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fasttext training')
    parser.add_argument('--input_file', type=str,
                        help='training file path (required)')
    parser.add_argument('--output', type=str,
                        help='output file path (required)')
    parser.add_argument('--lr', type=float, help='learning rate [0.05]')
    parser.add_argument('--lr_update_rate', type=int,
                        help='change the rate of updates for the learning rate [100]')
    parser.add_argument('--dim', type=int, help='size of word vectors [100]')
    parser.add_argument(
        '--ws', type=int, help='size of the context window [5]')
    parser.add_argument('--epoch', type=int, help='number of epochs [5]')
    parser.add_argument('--min_count', type=int,
                        help='minimal number of word occurences [5]')
    parser.add_argument('--neg', type=int,
                        help='number of negatives sampled [5]')
    parser.add_argument('--word_ngrams', type=int,
                        help='max length of word ngram [1]')
    parser.add_argument('--loss', type=str,
                        choices=['ns', 'hs', 'softmax'], help='loss function {ns, hs, softmax} [ns]')
    parser.add_argument('--bucket', type=int,
                        help='number of buckets [2000000]')
    parser.add_argument('--minn', type=int,
                        help='min length of char ngram [3]')
    parser.add_argument('--maxn', type=int,
                        help='max length of char ngram [6]')
    parser.add_argument('--thread', type=int, help='number of threads [12]')
    parser.add_argument('--t', type=float, help='sampling threshold [0.0001]')
    parser.add_argument('--silent', type=int,
                        help='disable the log output from the C++ extension [1]')
    parser.add_argument('--encoding', type=str,
                        help='specify input_file encoding [utf-8]')
    args = parser.parse_args()

    if args.sg == 1:
        fasttext.skipgram(args.input_file, args.target)
    elif args.sg == 0:
        fasttext.cbow(args.input_file, args.target)
    else:
        raise Exception("Pass either sg '1' or sg '0'")
