import itertools as it
from collections import deque
import random
import spacy


class Racoon:

    def __init__(self, window=None, **kwargs):
        self.window = window

    @staticmethod
    def sort(tokens, tdists, start):
        tokens = sorted(tokens, key=lambda t: abs(t.i - start.i))
        tokens = sorted(tokens, key=lambda t: tdists[t.i])
        return tokens

    def context(self, start):
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
        tokens = self.sort(tokens, tdists, start)
        return tokens

    def apply_context(self, context):
        context = (c.text for c in context)
        return it.islice(context, self.window)

    def tokencontext(self, doc):
        for token in doc:
            context = self.context(token)
            context = self.apply_context(context)
            yield token.text, list(context)

    def pairs(self, doc):
        for token, context in self.tokencontext(doc):
            yield from zip(it.repeat(token), context)

    def lines(self, doc):
        for token, context in self.tokencontext(doc):
            yield list(it.chain([token], context))


class POSRacoon(Racoon):

    @staticmethod
    def fine(token):
        return token.tag_

    @staticmethod
    def coarse(token):
        return token.pos_

    def __init__(self, fine=False, **kwargs):
        super().__init__(**kwargs)
        self.pos = (self.fine if fine else self.coarse)

    def apply_context(self, context):
        context = (f'{c.text}/{self.pos(c)}' for c in context)
        return it.islice(context, self.window)


class MinRacoon(Racoon):

    @staticmethod
    def sort(tokens, tdists, start):
        def key(t): return min(tdists[t.i], abs(t.i - start.i))
        return sorted(tokens, key=key)


class Squirrel(Racoon):

    @staticmethod
    def sort(tokens, tdists, start):
        tokens = [(t, tdists[t.i]) for t in tokens]
        random.shuffle(tokens)
        return sorted(tokens, key=lambda t: t[1])

    def apply_context(self, context):
        context = (f'{c.text}/{dist}' for c, dist in context)
        return context
