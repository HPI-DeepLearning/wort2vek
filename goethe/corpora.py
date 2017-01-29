import os


class Corpus:
    def __init__(self, path, use_tokens=True):
        """Pass path to corpus. Expects following structure:
        path/to/my/corpus
            corpus.txt
            corpus.tokens.txt
        """
        self.path = os.path.normpath(path)
        self.use_tokens = use_tokens

    def __iter__(self):
        """Return generator yielding either tokens or sentences.
        """
        return self.tokens if self.use_tokens else self.sents

    @property
    def sents(self):
        """Yield sentences from file.
        """
        path = self.file_path(use_tokens=False)
        with open(path) as f:
            yield from (l.strip() for l in f)

    @property
    def tokens(self):
        """Yield from file a list of tokens per sentence.
        """
        path = self.file_path(use_tokens=True)
        with open(path) as f:
            yield from (l.strip().split() for l in f)

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
