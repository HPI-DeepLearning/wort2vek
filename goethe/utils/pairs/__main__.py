import itertools as it
import functools as ft
import multiprocessing as mp
import random
import argparse
import sys
from tqdm import tqdm
import spacy
from .squirrel import Squirrel
from .racoon import Racoon, POSRacoon
from ..utils import args_to_kwargs, chunks


LANG = 'de'
BATCH_SIZE = 1000
N_THREADS = 4
PROCESSES = 16


methods = {'squirrel': Squirrel,
           'racoon': Racoon,
           'posracoon': POSRacoon}


def pairs_for_lines(lines, method):
    nlp = spacy.load(LANG)
    docs = nlp.pipe(lines, batch_size=BATCH_SIZE, n_threads=N_THREADS)
    return list(it.chain.from_iterable(method.pairs(doc) for doc in docs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create pairs file')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-m', '--method',
                        choices=methods.keys(), required=True)
    parser.add_argument('--max_level', type=int, default=3)
    parser.add_argument('--window', type=int)
    parser.add_argument('--fine', action='store_true')

    args = parser.parse_args()
    kwargs = args_to_kwargs(args)
    method = methods[args.method.lower()](**kwargs)

    with open(args.input) as inf, open(args.output, 'w') as outf, mp.Pool(8) as pool:
        lines = [l.strip() for l in inf]
        for pairs in pool.map(ft.partial(pairs_for_lines, method=method),
                              chunks(lines, PROCESSES),
                              chunksize=1):
            outputlines = [' '.join(it.chain([token], context)) + '\n'
                           for token, context in pairs]
            if args.output:
                outf.writelines(outputlines)
            else:
                sys.stdout.writelines(outputlines)
