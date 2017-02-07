import os
from .cleaner import Cleaner


class LeipzigCleaner(Cleaner):
    """Path should contain the data in files whose names contain `-sentences`.
       These files can be located in any subdirectory of the given path.
    """
    def __iter__(self):
        """Return iterator of cleaned sentences.
        """
        for file in self.sentence_files():
            with open(file) as f:
                yield from (self.clean_line(l) for l in f.readlines())

    @staticmethod
    def clean_line(line):
        """Apply to each line found in documents.
        """
        return line.split('\t')[1].strip()

    def sentence_files(self):
        """Yield a list of sentence txt files.
        """
        files = (os.path.join(dirname, f)
                 for dirname, _, fnames in os.walk(self.path)
                 for f in fnames)
        for f in files:
            if '-sentences' in f:
                yield f
