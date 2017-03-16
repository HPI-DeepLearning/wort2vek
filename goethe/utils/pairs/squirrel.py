from collections import deque
import random
import itertools as it


def word_prob(word):
    return 1


def dep_prob(dep_name):
    return 1


def should_sample(word, level):
    level_prob = 1 / level
    word_prob = word_prob(word)
    dep_prob = dep_prob(word.dep_)
    return random.random() < level_prob


def word_context(word, max_level):
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
        if level < max_level:
            add_to_queue(candidate, level + 1)


def squirrel(doc, max_level=3):
    for word in doc:
        for ctx in word_context(word, max_level):
            yield (word.text, ctx.text)
