import os

import spacy

LANG = 'de'


class Cleaner:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        """Return iterator of cleaned sentences.
        """
        raise NotImplementedError

    def write(self, write_path, delete=False):
        """For a path 'corpus.txt' write:
            corpus/
                corpus.txt
                corpus.tokens.txt

        Where 'corpus.txt' is one sent per line, spaces left in place,
        and 'corpus.tokens.txt' is one sent per line, token separated by space.
        """
        folder, file, tokens = self._write_paths(write_path)
        # Create corpus folder
        os.makedirs(folder, exist_ok=True)
        # Write sentence file
        with open(file, 'w') as f:
            f.writelines('%s\n' % l for l in self)
        # Write tokens file
        nlp = spacy.load(LANG)
        with open(file, 'w') as f:
            f.writelines(' '.join(str(token) for token in doc) + '\n'
                         for doc in nlp.pipe(self))

        # Remove uncleaned data
        if delete:
            if os.path.isfile(self.path):
                os.remove(self.path)
            elif os.path.isdir(self.path):
                os.rmdir(self.path)

    @staticmethod
    def _write_paths(path):
        """Return write paths for folder, file, and tokens.

        Example for input 'abc/def.txt':
            'abc/def', 'abc/def/def.txt', 'abc/def/def.tokens.txt'
        """
        # Example: abc/def.txt
        name = os.path.splitext(os.path.basename(path))[0]  # def
        folder = os.path.splitext(path)[0]  # abc/def/
        file = f'{name}.txt'  # abc/def.txt
        tokens = f'{name}.tokens.txt'  # abc/def.tokens.txt
        return folder, os.path.join(folder, file), os.path.join(folder, tokens)
