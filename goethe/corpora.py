import os
from . import util
import itertools as it

class Corpus:
    def __init__(self, path, limit=None):
        """Pass path to corpus. Expects following structure:
        path/to/my/corpus
            corpus.txt
            corpus.tokens.txt
        """
        self.path = os.path.normpath(path)
        self.limit = limit

    def __iter__(self):
        """Return generator yielding tokens.
        """
        return self.tokens()

    def sents(self):
        """Yield sentences from file.
        """
        path = self.file_path(use_tokens=False)
        with open(path) as f:
            yield from (l.strip() for l in f)

    def tokens(self):
        """Yield from file a list of tokens per sentence.
        """
        path = self.file_path(use_tokens=True)
        with open(path) as f:
            tokens = [line.strip().split() for line in f]
            yield from limit_iter(tokens) if self.limit else tokens

    def limit_iter(self, iterator):
        """Return iterator that yields self.limit elements of the passed
           iterator.
        """
        return it.islice(iterator, self.limit)

    def file_path(self, use_tokens):
        """Return path to either sentence or token file.
        Example:
            'abc/def/corpus'
                --> 'abc/def/corpus/corpus.txt'
                --> 'abc/def/corpus/corpus.tokens.txt'
        """
        corpus_name = os.path.basename(self.path)
        file = ('%s.tokens.txt' if use_tokens else '%s.txt') % corpus_name
        return os.path.join(self.path, file)

    def random(self, k):
        """Randomly select a list of k token lists.
        (Will load k elements into memory!)
        """
        return util.rsample(self.tokens(), k)
