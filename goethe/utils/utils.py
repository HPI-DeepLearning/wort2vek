import random
<<<<<<< HEAD
import sys
import contextlib
import itertools
=======
import contextlib
import sys
>>>>>>> 259f096f3c062a38db6b32273f72d8a332248f8c


def rsample(iterator, k):
    """Reservoir-sample k elements from iterator.
    Load and shuffle k elements in memory.

    http://propersubset.com/2010/04/choosing-random-elements.html
    """
    sample = []
    for n, item in enumerate(iterator):
        if len(sample) < k:
            sample.append(item)
        else:
            r = random.randint(0, n)
            if r < k:
                sample[r] = item

    random.shuffle(sample)
    return sample


def args_to_kwargs(args):
    return {k: v
            for k, v in vars(args).items()
            if v is not None}


def chunks(l, chunks):
    """Yield `chunks` successive chunks from l."""
    n = int(len(l) / chunks)
    for i in range(0, len(l), n):
        yield l[i:i + n]


def ichunks(iterable, n):
    it = iter(iterable)
    while True:
        chunk = itertools.islice(it, n)
        if not chunk:
            return
        yield chunk


@contextlib.contextmanager
def smart_open(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'w')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()
