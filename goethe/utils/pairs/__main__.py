import itertools as it
import functools as ft
import multiprocessing as mp
import random
import argparse
import sys
from tqdm import tqdm
import spacy
# from .squirrel import Squirrel
from .methods import Racoon, POSRacoon, MinRacoon, Squirrel
from ..utils import args_to_kwargs, chunks


LANG = 'de'
BATCH_SIZE = 10000
N_THREADS = 2
PROCESSES = 16


methods = {'squirrel': Squirrel,
           'racoon': Racoon,
           'posracoon': POSRacoon,
           'minracoon': MinRacoon}


def contexts_for_lines(lines, method):
    nlp = spacy.load(LANG)
    docs = nlp.pipe(lines, batch_size=BATCH_SIZE, n_threads=N_THREADS)
    return list(it.chain(method.lines(doc) for doc in docs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create pairs file')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-m', '--method',
                        choices=methods.keys(), required=True)
    # parser.add_argument('--max_level', type=int, default=3)
    parser.add_argument('--window', type=int)
    parser.add_argument('--fine', action='store_true')

    args = parser.parse_args()
    kwargs = args_to_kwargs(args)
    method = methods[args.method.lower()](**kwargs)

    with open(args.input) as inf, open(args.output, 'w') as outf, mp.Pool(8) as pool:
        lines = [l.strip() for l in inf]
        for contexts in pool.map(ft.partial(contexts_for_lines, method=method),
                                 chunks(lines, PROCESSES),
                                 chunksize=1):
            outputlines = (' '.join(context) + '\n'
                           for context in contexts)
            outf.writelines(outputlines)
