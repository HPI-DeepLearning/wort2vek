import itertools as it
import multiprocessing as mp
import random
import argparse
from tqdm import tqdm
import spacy
from squirrel import Squirrel


LANG = 'de'
BATCH_SIZE = 10000
N_THREADS = 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create file of word context pairs.')
    parser.add_argument('input', metavar='FILE')
    parser.add_argument('--output', metavar='FILE')
    parser.add_argument('--max_level', metavar='N', type=int, default=3)
    args = parser.parse_args()

    input_file = args.input

    nlp = spacy.load(LANG, vectors=False)

    def count_lines(filename):
        for i, _ in enumerate(open(filename)):
            pass
        return i + 1

    with open(input_file, 'r') as f:
        lines = (line.strip() for line in f)
        docs = nlp.pipe(lines, batch_size=BATCH_SIZE, n_threads=N_THREADS)
        # docs = tqdm(docs, total=count_lines(input_file))
        squirrel = Squirrel(args.max_level)
        pairs = (pair
                 for doc in docs
                 for pair in squirrel.pairs(doc))
        out = (word + ' ' + ctx for word, ctx in pairs)

        if args.output:
            with open(args.output, 'w') as f:
                f.writelines(l + '\n' for l in out)
        else:
            for l in out:
                print(l)
