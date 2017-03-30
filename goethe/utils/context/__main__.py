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


def contexts_for_sents(sents, method, pairs=False):
    contexts = method.pairs if pairs else method.lines
    nlp = spacy.load(LANG)
    docs = nlp.pipe(sents, batch_size=BATCH_SIZE, n_threads=N_THREADS)
    for doc in docs:
        yield from contexts(doc)


def main(args):
    method = methods[args.method.lower()](**args_to_kwargs(args))
    with open(args.input) as inf, smart_open(args.output) as outf:
        lines = (l.rstrip('\n') for l in inf)
        contexts = contexts_for_sents(lines, method, pairs=args.pairs)
        outputlines = (' '.join(c) + '\n' for c in contexts)
        outf.writelines(outputlines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create context file')
    io = parser.add_argument_group('Input/Output')
    io.add_argument('-i', '--input', required=True)
    io.add_argument('-o', '--output')
    method = parser.add_argument_group('Method')
    method.add_argument('-m', '--method', required=True,
                        choices=methods.keys())
    pairslists = method.add_mutually_exclusive_group(required=True)
    pairslists.add_argument('-p', '--pairs', action='store_true',
                            help='Per line: One \'word context\' pair')
    pairslists.add_argument('-l', '--lists', action='store_true',
                            help='Per line: \'word c1 c2 c3 ...\'')
    margs = parser.add_argument_group('Method arguments')
    margs.add_argument('-max_level', type=int, default=3)
    margs.add_argument('-window', type=int)
    margs.add_argument('-fine', action='store_true')
    args = parser.parse_args()
    main(args)
