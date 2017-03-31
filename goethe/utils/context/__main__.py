import itertools as it
import functools as ft
import multiprocessing as mp
import random
import argparse
import spacy
from .methods import *
from ..utils import args_to_kwargs, chunks, smart_open, iterlen

description = """
Generate context file. Takes a file with one sentence per line as input.
Returns contexts either in `pairs` format or in `lines` format.
"""

LANG = 'de'
BATCH_SIZE = 10000
N_THREADS = 8

methods = {'ttraverse': TreeTraverse,
           'torder': TreeOrder,
           'tposorder': POSTreeOrder,
           'tminorder': MinTreeOrder,
           'tshuffleorder': ShuffleTreeOrder,
           'linear': Linear,
           'levygoldberg': LevyGoldberg}


def main(args):
    method = methods[args.method.lower()](**args_to_kwargs(args))
    contextfunc = method.pairs if args.pairs else method.lines

    nlp = spacy.load(LANG)

    with open(args.input) as inf, smart_open(args.output) as outf:
        sents = (l.rstrip('\n') for l in inf)
        docs = nlp.pipe(sents, batch_size=BATCH_SIZE, n_threads=N_THREADS)
        for i, doc in enumerate(docs):
            contexts = contextfunc(doc)
            contexts = (' '.join(c) + '\n' for c in contexts)
            if args.numbers:
                contexts = (f'{i}\t{c}' for c in contexts)
            outf.writelines(contexts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    io = parser.add_argument_group('Input/Output')
    io.add_argument('-i', '--input', required=True)
    io.add_argument('-o', '--output')
    parser.add_argument('-n', '--numbers', action='store_true',
                        help='Add line numbers')
    method = parser.add_argument_group('Method')
    method.add_argument('-m', '--method', required=True,
                        choices=methods.keys())
    pairslists = method.add_mutually_exclusive_group(required=True)
    pairslists.add_argument('-p', '--pairs', action='store_true',
                            help='Pairs format: per line one \'word context\' pair')
    pairslists.add_argument('-l', '--lists', action='store_true',
                            help='Lines format: per line \'word c1 c2 c3 ...\'')
    margs = parser.add_argument_group('Method arguments')
    margs.add_argument('-max_level', type=int, default=3)
    margs.add_argument('-window', type=int)
    margs.add_argument('-fine', action='store_true')
    args = parser.parse_args()
    main(args)
