import os
import spacy

LANG = 'de'
BATCH_SIZE = 10000
N_THREADS = 4


class Cleaner:
    def __init__(self, path):
        self.path = path
        self._nlp = None

    def __iter__(self):
        """Return iterator of cleaned sentences.
        """
        yield from (line.strip() for line in open(self.path))

    def write(self, write_path, delete=False):
        """For a path 'pat/to/corpus' write:
            path/to/corpus/
                corpus.txt
                corpus.tokens.txt
        where 'corpus.txt' is one sent per line, spaces left in place,
        and 'corpus.tokens.txt' is one sent per line, token separated by space.
        """
        folder, file, tokens = self._write_paths(write_path)
        # Create corpus folder
        os.makedirs(folder, exist_ok=True)

        # Write tokens and sentences files
        with open(file, 'w') as f_sents, open(tokens, 'w') as f_tokens:
            for doc in self.tokenized_sents():
                for sent_doc in doc.sents:
                    self.write_sentence(f_sents, sent_doc.string)
                    self.write_tokens(f_tokens, sent_doc)

        # Remove uncleaned data
        if delete:
            if os.path.isfile(self.path):
                os.remove(self.path)
            elif os.path.isdir(self.path):
                os.rmdir(self.path)

    def write_sentence(self, file, sentence):
        file.write('%s\n' % sentence.strip())

    def write_tokens(self, file, tokens):
        token_line = ' '.join(str(token) for token in tokens)
        file.write('%s\n' % token_line)

    def tokenized_sents(self):
        """Return an iterator with SpaCy docs.
        The docs' tokens can be iterated:
        Example:
            [token
             for doc in self.tokenized_sents()
             for token in doc]
        """
        return self.nlp.pipe(self,
                             batch_size=BATCH_SIZE,
                             n_threads=N_THREADS)

    @property
    def nlp(self):
        if not self._nlp:
            self._nlp = spacy.load(LANG)
        return self._nlp

    @staticmethod
    def _write_paths(path):
        """Return write paths for folder, file, and tokens.
        Example for input 'abc/def/':
            'abc/def', 'abc/def/def.txt', 'abc/def/def.tokens.txt'
        """
        # Example: abc/def/
        path = os.path.normpath(path)  # abc/def
        name = os.path.splitext(os.path.basename(path))[0]  # def
        folder = os.path.splitext(path)[0]  # abc/def/
        file = '%s.txt' % name  # abc/def.txt
        tokens = '%s.tokens.txt' % name  # abc/def.tokens.txt
        return folder, os.path.join(folder, file), os.path.join(folder, tokens)
