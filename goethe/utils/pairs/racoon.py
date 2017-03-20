import itertools as it
from collections import namedtuple, deque
import random
import spacy


class Racoon:

    def __init__(self, window=5, **kwargs):
        self.window = window

    @staticmethod
    def ptsort(start):
        tdists = [0] * len(start.doc)
        queue = deque([start])
        seen = {start}

        def neighbors(token):
            is_head = token.dep_ == 'ROOT'
            return it.chain(token.children, [] if is_head else [token.head])

        while queue:
            t = queue.popleft()
            nbrs = [n for n in neighbors(t)
                    if n not in seen]
            for n in nbrs:
                tdists[n.i] = tdists[t.i] + 1
            seen.update(nbrs)
            queue.extend(nbrs)

        tokens = (t for t in start.doc if t is not start)
        tokens = sorted(tokens, key=lambda t: abs(t.i - start.i))
        tokens = sorted(tokens, key=lambda t: tdists[t.i])
        return tokens

    def pairs(self, doc):

        def per_token(token):
            context = self.ptsort(token)
            pairs = zip(it.repeat(token), context)
            return it.islice(pairs, self.window)

        return it.chain.from_iterable(per_token(t) for t in doc)


class POSRacoon(Racoon):

    def __init__(self, fine=False, **kwargs):
        super().__init__(**kwargs)
        self.pos = ((lambda t: t.tag_)
                    if fine else
                    (lambda t: t.pos_))

    def pairs(self, doc):
        pairs = super().pairs(doc)
        return ((t.text, f'{c.text}/{self.pos(c)}')
                for t, c in pairs)
