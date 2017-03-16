import os
import argparse
import subprocess
import itertools as it

# name =$2
# cv = "$name.cv.txt"
# wv = "$name.wv.txt"
# . / count_and_filter - train $1 - cvocab $cv - wvocab $wv - min - count 100
# . / word2vecf - train $1 - wvocab $wv - cvocab $cv - output "$name.vec" - dumpcv "$name.cv.vec" - size 300 - negative 15 - threads 10 - iters 10

WORD2VEC_PATH = 'word2vecf'
COUNT_AND_FILTER = 'count_and_filter'
WORD2VECF = 'word2vecf'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word2vecf training')
    parser.add_argument('input', help='pairs file')
    parser.add_argument('output', help='name of outputted model')
    parser.add_argument('-m', '--mincount', help='min count [100]', type=int, default=100)
    parser.add_argument('-s', '--size', help='word vector size [300]', type=int, default=300)
    parser.add_argument('-n', '--negative',
                        help='number of negative samples [15]', type=int, default=15)
    parser.add_argument('-t', '--threads',
                        help='number of threads [10]', type=int, default=10)
    parser.add_argument(
        '-i', '--iters', help='number of iterations [10]', type=int, default=10)
    args = parser.parse_args()

    word_vocab = f'{args.output}.cv.txt'
    context_vocab = f'{args.output}.wv.txt'

    subprocess.call([os.path.join(WORD2VEC_PATH, COUNT_AND_FILTER),
                     *it.chain.from_iterable({
                         '-train':       args.input,
                         '-wvocab':      f'{args.output}.cv.txt',
                         '-cvocab':      f'{args.output}.wv.txt',
                         '-min-count':   str(args.mincount),
                     }.items())])

    subprocess.call([os.path.join(WORD2VEC_PATH, WORD2VECF),
                     *it.chain.from_iterable({
                         '-train':       args.input,
                         '-wvocab':      f'{args.output}.cv.txt',
                         '-cvocab':      f'{args.output}.wv.txt',
                         '-output':      f'{args.output}.vec',
                         '-dumpcv':      f'{args.output}.cv.vec',
                         '-size':        str(args.size),
                         '-negative':    str(args.negative),
                         '-threads':     str(args.threads),
                         '-iters':       str(args.iters),
                     }.items())])
