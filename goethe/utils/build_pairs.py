import sys
import spacy
from collections import deque
from tqdm import tqdm
import itertools as it
from random import random
import argparse

LANG = 'de'
BATCH_SIZE = 10000
N_THREADS = 4

parser = argparse.ArgumentParser(description='Create file of word context pairs.')
parser.add_argument('input', metavar='FILE')
parser.add_argument('--output', metavar='FILE')
parser.add_argument('--max_level', metavar='N', type=int, default=3)
args = parser.parse_args()

MAX_LEVEL = args.max_level
input_file = args.input

nlp = spacy.load(LANG, vectors=False)

def hmean(*values):
    return len(values) / sum(1 / x for x in values)

pos_high_p = ['VERB', 'NOUN']
pos_middle_p = ['ADV', 'PPER', 'ADJ']

def pos_p(pos):
    if pos in pos_high_p:
        return 0.8
    elif pos in pos_middle_p:
        return 0.6
    else:
        return 0.4

def should_sample(word, level):
    level_prob = 1 / level
    # Todo: calc pos and idf prob
    # pos_prob = pos_p(word.pos_)
    # idf_prob = 1
    # prob = hmean(level_prob, pos_prob, idf_prob)
    return random() < level_prob

def sample_context(word):
    candidates = deque()
    seen = set([word])

    def add_to_queue(word, level):
        for w in it.chain(word.children, [word.head]):
            if not w in seen:
                seen.add(w)
                candidates.append((level, w))

    add_to_queue(word, 1)
    while candidates:
        level, candidate = candidates.popleft()
        if should_sample(candidate, level):
            yield candidate
        if level < MAX_LEVEL:
            add_to_queue(candidate, level + 1)

def doc2pairs(doc):
    for word in doc:
        for ctx in sample_context(word):
            yield (word.text, ctx.text)

def count_lines(filename):
    count = 0
    for _ in open(filename):
        count += 1
    return count

with open(input_file, 'r') as f:
    lines = (line.strip() for line in f)
    docs = nlp.pipe(lines, batch_size=BATCH_SIZE, n_threads=N_THREADS)
    docs = tqdm(docs, total=count_lines(input_file))
    pairs = (pair for doc in docs for pair in doc2pairs(doc))
    out = (word + ' ' + ctx for word, ctx in pairs)

    if args.output:
        with open(args.output, 'w') as f:
            for l in out:
                f.write(l + '\n')
    else:
        for l in out: print(l)
