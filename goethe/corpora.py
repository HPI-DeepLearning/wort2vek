import os
import collections


class LeipzigCorpus:
    """Iterate over Leipzig Corpus (part of Projekt Deutscher Wortschatz).
    """

    def __init__(self, dirname, lang='deu', corpus_name=None, max_sentences=None):
        self.dirname = dirname
        self.lang = lang
        self.corpus_name = corpus_name
        self.max_sentences = max_sentences

    def __iter__(self):
        for i, s in enumerate(self.sentences()):
            if self.max_sentences and i > self.max_sentences:
                raise StopIteration
            yield s

    def raw_sentences(self):
        """Find all sentence files in 'dirname' and iterate over lines
        returning sentences only (without leading numbers).

        Assumes file tree like following:
            dirname/
                deu_news_2015_3M/
                    deu_news_2015_3M-sentences.txt
                    ...
                deu_wikipedia_2014_3M/
                    deu_wikipedia_2014_3M-sentences.txt
                    ...
                ...
        """
        for corpus in os.listdir(self.dirname):
            # Select only corpora with 'self.lang' language
            if not corpus.startswith(self.lang):
                continue

            if self.corpus_name and corpus != self.corpus_name:
                continue

            for fname in os.listdir(os.path.join(self.dirname, corpus)):
                if not fname.endswith('sentences.txt'):
                    continue
                sentences_file = os.path.join(self.dirname, corpus, fname)
                for line in open(sentences_file):
                    # Lines are of form: 'LineNumber\tActualSentence\n'
                    yield line.split('\t')[1].strip()

    def sentences(self):
        for s in self.raw_sentences():
            yield s.split()

    def words(self):
        for sentence in self.__iter__():
            for word in sentence:
                yield word
