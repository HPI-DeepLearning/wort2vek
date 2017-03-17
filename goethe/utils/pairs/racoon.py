import itertools as it
from collections import namedtuple, deque
import random

import spacy


class Racoon:

    @classmethod
    def ptsort(cls, start):
        tdists = [0] * len(start.doc)
        queue = deque([start])
        seen = {start}

        def neighbors(token):
            is_head = token.dep_ == 'ROOT'
            return it.chain(token.children, [] if is_head else [token.head])

        while queue:
            t = queue.popleft()
            nbrs = [n for n in neighbors(t) if n not in seen]
            for n in nbrs:
                tdists[n.i] = tdists[t.i] + 1
            queue.extend(nbrs)
            seen.add(t)

        tokens = (t for t in start.doc if t is not start)
        tokens = sorted(tokens, key=lambda t: abs(t.i - start.i))
        tokens = sorted(tokens, key=lambda t: tdists[t.i])
        return tokens

    @classmethod
    def ptsort_pairs(cls, doc):
        context_size = random.randint(1, len(doc))

        def per_token(token):
            pairs = zip(it.repeat(token), cls.ptsort(token))
            return it.islice(pairs, context_size)

        for t in doc:
            yield from per_token(t)

    @classmethod
    def pairs(cls, corpus):
        nlp = spacy.load('de')
        docs = nlp.pipe(corpus.sents(), batch_size=100_000, n_threads=8)
        for d in docs:
            yield from cls.ptsort_pairs(d)
