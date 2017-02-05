from .cleaner import Cleaner


class WikipediaCleaner(Cleaner):
    """Iterates over Wikipedia corpus file and filteres out lines with a length
       less than `min_length`. We use the corpus from:
       http://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/
       The passed corpus file must be cleaned with the Per scripts provided on
       this website (we used the -nomath and -notables options).
    """

    def __init__(self, path, min_length=6):
        super().__init__(path)
        self.min_length = min_length

    def __iter__(self):
        """Return iterator of filtered sentences.
        """
        for line in open(self.path):
            if len(line.split()) >= self.min_length:
                yield line
