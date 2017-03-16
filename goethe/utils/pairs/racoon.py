import itertools as it
from collections import namedtuple, deque
import random

CollapsedToken = namedtuple('CollapsedToken', 'text dep_')


def collapse(head, token):
    return CollapsedToken(text=token.text,
                          dep_=f'{head.dep_}_{head.text}')


class ParseTreeError(Exception):
    pass


def prefix_dep(nodes):
    return [f'{n.dep_}/{n.text}' for n in nodes]


def handle_modifier(t):
    """Find and return noun kernel element (nk)
    or return node itself.
    """
    nk = next((c for c in t.children if c.dep_ == 'nk'), None)
    if not nk:
        raise ParseTreeError
    else:
        return [t, collapse(t, nk)]
    # return collapse(nk, n) if nk else t


def handle_coordinative_conjunction(t):
    """Find and return conjunct (cj).
    """
    cj = next((c for c in t.children if c.dep_ == 'cj'))
    if not cj:
        raise ParseTreeError
    else:
        return [t,  collapse(t, cj)]


def handle(token):
    def default(n):
        return [n]

    def skip(n):
        return []

    return ({
        'punct': skip,
        'op': default,
        'mo': handle_modifier,
        'cd': handle_coordinative_conjunction,
    }.get(token.dep_) or default)(token)


def collect_children(token):
    childs = it.chain.from_iterable(handle(t) for t in token.children)
    deps, texts = zip(*((c.dep_, c.text) for c in childs if c))
    deps_inverse = (f'{d}I' for d in deps)
    directed = ((head, f'{target}/{dep}')
                for head, dep, target in zip(it.repeat(token), deps, texts))
    inversed = ((target, f'{head}/{dep}I')
                for head, dep, target in zip(it.repeat(token), deps, texts))
    return list(it.chain(directed, inversed))


def pairs(doc):
    root = root_token(doc)
    return collect_children(root)


def print_tree(root):
    # root = root_token(doc)
    neighbors = sorted(list(root.children) + [root], key=lambda n: n.i)
    context = [f'{n.dep_}/{n}' for n in neighbors if not n == root]
    print(f'{root} --> {context}')
    for n in neighbors:
        if not n == root:
            print_tree(n)
    # [print_tree(n) for n in neighbors if n.is_ancestor]


def root_token(doc):
    n = next(iter(doc))
    while n.dep_ != 'ROOT':
        n = n.head
    return n


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


def ptsort(start):
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


def ptsort_pairs(doc):
    context_size = random.randint(1, len(doc) - 1)

    def per_token(token):
        pairs = zip(it.repeat(token), ptsort(token))
        return it.islice(pairs, context_size)

    return it.chain.from_iterable(per_token(t) for t in doc)
