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
    parser.add_argument('-m', '--mincount', help='min count', default=100)
    parser.add_argument('-s', '--size', help='word vector size', default=300)
    parser.add_argument('-n', '--negative',
                        help='number of negative samples', default=15)
    parser.add_argument('-t', '--threads',
                        help='number of threads', default=10)
    parser.add_argument(
        '-i', '--iters', help='number of iterations', default=10)
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
                         '-size':        args.size,
                         '-negative':    args.negative,
                         '-threads':     args.threads,
                         '-iters':       args.iters,
                     }.items())])
