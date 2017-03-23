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
from .window import Window
from ..utils import args_to_kwargs, chunks

LANG = 'de'
BATCH_SIZE = 1000
N_THREADS = 2
PROCESSES = 1


methods = {'squirrel': Squirrel,
           'racoon': Racoon,
           'posracoon': POSRacoon,
           'window': Window}


def pairs_for_lines(lines, method):
    if method.PARSE_TREE:
        nlp = spacy.load(LANG)
    else:
        nlp = spacy.load(LANG, parser=False)
    docs = nlp.pipe(lines, batch_size=BATCH_SIZE, n_threads=N_THREADS)
    return list(it.chain.from_iterable(method.token_list(doc) for doc in docs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create pairs file')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output')
    parser.add_argument('-m', '--method',
                        choices=methods.keys(), required=True)
    parser.add_argument('--max_level', type=int, default=3)
    parser.add_argument('--window', type=int)
    parser.add_argument('--fine', action='store_true')

    args = parser.parse_args()
    kwargs = args_to_kwargs(args)
    method = methods[args.method.lower()](**kwargs)

    with open(args.input) as inf, mp.Pool(8) as pool:
        lines = [l.strip() for l in inf]
        for lists in pool.map(ft.partial(pairs_for_lines, method=method),
                              chunks(lines, PROCESSES),
                              chunksize=1):
            outputlines = [' '.join(it.chain([token], context)) + '\n'
                           for token, context in lists]
            if args.output:
                with open(args.output, 'w') as outf:
                    outf.writelines(outputlines)
            else:
                sys.stdout.writelines(outputlines)
