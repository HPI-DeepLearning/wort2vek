import itertools as it
import functools as ft
import multiprocessing as mp
import random
import argparse
import spacy
from .window import Window
from .squirrel import Squirrel
from .methods import Racoon, POSRacoon, MinRacoon, Squirrel
from ..utils import args_to_kwargs, chunks, smart_open, iterlen


LANG = 'de'
BATCH_SIZE = 10000
N_THREADS = 8

methods = {'squirrel': Squirrel,
           'racoon': Racoon,
           'posracoon': POSRacoon,
           'window': Window,
           'minracoon': MinRacoon}


def contexts_for_lines(lines, method):
    nlp = spacy.load(LANG)
    docs = nlp.pipe(lines, batch_size=BATCH_SIZE, n_threads=N_THREADS)
    for doc in docs:
        yield from method.lines(doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create pairs file')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-m', '--method', required=True,
                        choices=methods.keys())
    parser.add_argument('-o', '--output')
    parser.add_argument('--max_level', type=int, default=3)
    parser.add_argument('--window', type=int)
    parser.add_argument('--fine', action='store_true')

    args = parser.parse_args()
    kwargs = args_to_kwargs(args)
    method = methods[args.method.lower()](**kwargs)

    with open(args.input) as inf, smart_open(args.output) as outf:
        lines = (l.strip() for l in inf)
        contexts = contexts_for_lines(lines, method)
        outputlines = (' '.join(context) + '\n' for context in contexts)
        outf.writelines(outputlines)
