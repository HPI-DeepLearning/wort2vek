import sys
import random
import contextlib
import itertools as it


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


def chunks(lines, length, chunks):
    """Yield `chunks` successive chunks from iterable.
    """
    chunksize = int(length / chunks)
    iterator = iter(lines)
    for i in range(0, length, chunksize):
        yield list(it.islice(iterator, chunksize))


def iterlen(f):
    for i, _ in enumerate(f):
        pass
    return i + 1


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
