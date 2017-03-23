import random
import contextlib
import sys


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


@contextlib.contextmanager
def openfileorstdout(output=None):
    if output is None:
        f = sys.stdout
    else:
        f = open(output, 'w')
    try:
        yield f
    finally:
        if f is not sys.stdout:
            f.close()
