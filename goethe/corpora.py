import os


class LeipzigCorpus:
    """Iterate over Leipzig Corpus (part of Projekt Deutscher Wortschatz).
    """

    def __init__(self, dir_name, lang='deu', corpus_name=None,
                 max_sentences=None, name_filter=None, words=True):
        """Use 'condition' to filter names, e.g. 'Wikipedia'.
        """
        self.dir = dir_name
        self.lang = lang
        self.corpus_name = corpus_name
        self.max_sentences = max_sentences
        self.name_filter = name_filter
        self.words = words

    def __iter__(self):
        for i, s in enumerate(self.sentences()):
            if self.max_sentences and i >= self.max_sentences:
                break
            yield s

    def sentences(self):
        """Iterate over lines in files returning sentences.
        """
        files = self.sentence_files()
        for f in files:
            for line in open(f):
                # Lines are of form: 'LineNumber\tActualSentence\n'
                sent = line.split('\t')[1].strip()
                yield sent.split() if self.words else sent

    def sentence_files(self):
        """Find all sentence files in 'dirname'.
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
        files = []
        for corpus in os.listdir(self.dir):
            # Select only corpora with 'self.lang' language
            if not corpus.startswith(self.lang):
                continue

            if self.corpus_name and corpus != self.corpus_name:
                continue

            for f in os.listdir(os.path.join(self.dir, corpus)):
                if not f.endswith('sentences.txt'):
                    continue
                if self.name_filter and not self.name_filter(f):
                    continue
                sentences_file = os.path.join(self.dir, corpus, f)
                files.append(sentences_file)
        return files
