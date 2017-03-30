import itertools as it
from collections import deque
import random
import spacy


class ContextMethod:

    def __init__(self, window=None, **kwargs):
        self.window = window

    def tokenwise_context(self, doc):
        """Go over each word in `context` and get its text.
        """
        raise NotImplementedError

    def apply_context(self, token):
        """Apply to each context token.
        """
        return token.text

    def pairs(self, doc):
        """Yield context in pairs.
        """
        for token, context in self.tokenwise_context(doc):
            yield from zip(it.repeat(token), context)

    def lines(self, doc):
        """Yield context in lists with the first element being the word.
        """
        for token, context in self.tokenwise_context(doc):
            yield (token, *context)

    def tokenwise_context(self, doc):
        """Go over each word in `context` and get its text.
        """
        for token in doc:
            context = self.context(token)
            context = (self.apply_context(c) for c in context)
            context = it.islice(context, self.window)
            yield token.text, context


class TreeOrder(ContextMethod):

    def contextsort(self, tokens, tdists, start):
        """Sort a `start` words context `tokens` given the
        tree distances (`tdists`).
        """
        tokens = sorted(tokens, key=lambda t: abs(t.i - start.i))
        tokens = sorted(tokens, key=lambda t: tdists[t.i])
        return tokens

    def context(self, start):
        """Take `start` word and return iterable over its context.
        """
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
        tokens = self.contextsort(tokens, tdists, start)
        return tokens


class POSTreeOrder(TreeOrder):

    @staticmethod
    def fine(token):
        """Return a token's fine POS tag.
        """
        return token.tag_

    @staticmethod
    def coarse(token):
        """Return a token's coarse POS tag.
        """
        return token.pos_

    def __init__(self, fine=False, **kwargs):
        super().__init__(**kwargs)
        self.pos = (self.fine if fine else self.coarse)

    def apply_context(self, token):
        """Apply to each context token.
        """
        return f'{token.text}/{self.pos(token)}'


class MinTreeOrder(TreeOrder):

    def contextsort(self, tokens, tdists, stkrt):
        """Sort a `start` words context `tokens` given the
        tree distances (`tdists`). As key, use
        min(tree distance, sentence distance).
        """
        def key(t): return min(tdists[t.i], abs(t.i - start.i))
        tokens = sorted(tokens, key=key)
        return tokens


class ShuffleTreeOrder(TreeOrder):

    def contextsort(self, tokens, tdists, start):
        """Sort a `start` words context `tokens` given the
        tree distances (`tdists`). Shuffle tokens with equal tree distance.
        """
        tokens = [(t, tdists[t.i]) for t in tokens]
        random.shuffle(tokens)
        return sorted(tokens, key=lambda t: t[1])

    def apply_context(self, token):
        """Apply to each context token.
        """
        c, dist = token
        return f'{c.text}/{dist}'


class TreeTraverse(ContextMethod):

    def __init__(self, max_level, **kwargs):
        super().__init__(**kwargs)
        self.max_level = max_level

    def word_prob(self, word):
        return 1

    def dep_prob(self, dep_name):
        return 1

    def inv_level_prob(self, level):
        return 1 / level

    def word2vec_window_prob(self, level):
        return (self.max_level - level + 1) / self.max_level

    def keep(self, word, level):
        level_prob = self.word2vec_window_prob(level)
        # word_prob = self.word_prob(word)
        # dep_prob = self.dep_prob(word.dep_)
        return random.random() < level_prob

    def context(self, word):
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
            if self.keep(candidate, level):
                yield candidate
            if level < self.max_level:
                add_to_queue(candidate, level + 1)


class Linear(ContextMethod):

    def window_sort(self, tokens, n):
        i = n - 1
        j = n + 1
        l = []
        while i >= 0 or j < len(tokens):
            if i >= 0:
                l.append(tokens[i])
            if j < len(tokens):
                l.append(tokens[j])
            i -= 1
            j += 1
        return l

    def tokenwise_context(self, doc):
        """Go over each word in `context` and get its text.
        """
        for i, token in enumerate(doc):
            context = self.window_sort(doc, i)
            context = (self.apply_context(c) for c in context)
            context = it.islice(context, self.window)
            yield token.text, context
