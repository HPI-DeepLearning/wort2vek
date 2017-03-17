import itertools as it
import multiprocessing as mp
import random
import argparse
import sys
from tqdm import tqdm
import spacy
from .squirrel import Squirrel
from .racoon import Racoon


LANG = 'de'
BATCH_SIZE = 100_000
N_THREADS = 8

methods = {'squirrel': lambda args: Squirrel(args.max_level),
           'racoon': lambda args: Racoon()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create pairs file')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output')
    parser.add_argument(
        '-m', '--method', choices=methods.keys(), required=True)
    parser.add_argument('--max_level', metavar='N', type=int, default=3)
    args = parser.parse_args()

    nlp = spacy.load(LANG)

    with open(args.input) as inf:
        lines = (l.strip() for l in inf)
        docs = nlp.pipe(lines, batch_size=BATCH_SIZE, n_threads=N_THREADS)
        pairsgen = methods[args.method](args).pairs
        pairs = (f'{word} {context}\n'
                 for doc in docs
                 for word, context in pairsgen(doc))

        if args.output:
            with open(args.output, 'w') as outf:
                outf.writelines(pairs)
        else:
            sys.stdout.writelines(pairs)
