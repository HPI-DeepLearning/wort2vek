from collections import deque
import itertools as it
import multiprocessing as mp
import random
import argparse
from tqdm import tqdm
import spacy

LANG = 'de'
BATCH_SIZE = 10000
N_THREADS = 4


def pairs(method, docs, chunksize=10000):
    with mp.Pool(4) as pool:
        yield from pool.imap(method, docs, chunksize)


def word2vec(tokens, contextsize=5):

    def per_word_context_pairs(i):
        size = random.randint(1, contextsize)
        start = max(0, i - size)
        end = 1 + i + size
        contextleft = tokens[start: i]
        contextright = tokens[i + 1: end]
        return zip(it.repeat(tokens[i]), contextleft + contextright)

    return list(it.chain.from_iterable(per_word_context_pairs(i)
                                       for i, _ in enumerate(tokens)))


def word_prob(word):
    return 1


def dep_prob(dep_name):
    return 1


def should_sample(word, level):
    level_prob = 1 / level
    word_prob = word_prob(word)
    dep_prob = dep_prob(word.dep_)
    return random.random() < level_prob


def word_context(word):
    candidates = deque()
    seen = {word}

    def add_to_queue(word, level):
        for w in it.chain(word.children, [word.head]):
            if w not in seen:
                seen.add(w)
                candidates.append((level, w))

    add_to_queue(word, 1)
    while candidates:
        level, candidate = candidates.popleft()
        if should_sample(candidate, level):
            yield candidate
        if level < MAX_LEVEL:
            add_to_queue(candidate, level + 1)


def squirrel(doc):
    for word in doc:
        for ctx in word_context(word):
            yield (word.text, ctx.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create file of word context pairs.')
    parser.add_argument('input', metavar='FILE')
    parser.add_argument('--output', metavar='FILE')
    parser.add_argument('--max_level', metavar='N', type=int, default=3)
    args = parser.parse_args()

    MAX_LEVEL = args.max_level
    input_file = args.input

    nlp = spacy.load(LANG, vectors=False)

    def count_lines(filename):
        count = 0
        for _ in open(filename):
            count += 1
        return count

    with open(input_file, 'r') as f:
        lines = (line.strip() for line in f)
        docs = nlp.pipe(lines, batch_size=BATCH_SIZE, n_threads=N_THREADS)
        docs = tqdm(docs, total=count_lines(input_file))
        pairs = (pair for doc in docs for pair in squirrel(doc))
        out = (word + ' ' + ctx for word, ctx in pairs)

        if args.output:
            with open(args.output, 'w') as f:
                for l in out:
                    f.write(l + '\n')
        else:
            for l in out:
                print(l)
