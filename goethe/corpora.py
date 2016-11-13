import os


class LeipzigCorpus:
    """Iterate over Leipzig Corpus (part of Projekt Deutscher Wortschatz).
    """

    def __init__(self, dirname, lang='deu'):
        self.dirname = dirname
        self.lang = lang

    def __iter__(self):
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
            for fname in os.listdir(os.path.join(self.dirname, corpus)):
                if not fname.endswith('sentences.txt'):
                    continue
                sentences_file = os.path.join(self.dirname, corpus, fname)
                for line in open(sentences_file):
                    # Lines are of form: 'LineNumber\tActualSentence\n'
                    yield line.split('\t')[1].strip()
