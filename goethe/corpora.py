import os
from nltk import word_tokenize
import numpy as np
from collections import Counter
from itertools import chain

class LineNumbers:
    """Iterate over a file of sentences, where each word is represented
       by a number.
    """

    def __init__(self, filename):s
        self.filename = filename

    def sentences(self):
        for line in open(self.filename):
            yield map(int, line.split())

    def ngrams(self, n):
        for numbers in self.sentences():
            for i in range(0, len(numbers) - n + 1):
                yield numbers[i:i+n]

    def word_index(self, filename, nb_words=None):
        """ word index starts at 1, 0 is reserved for words with a rank higher
            than nb_words
        """
        with open(filename, 'r') as f:
            index2word = f.readlines()
            if nb_words:
                index2word = index2word[:nb_words]
            index2word = ['unknown'] + index2word
            word2index = {word: index for index, word in enumerate(index2word)}
        return index2word, word2index

class LeipzigCorpus:
    """Iterate over Leipzig Corpus (part of Projekt Deutscher Wortschatz).
    """

    def __init__(self, dir_name, lang='deu', corpus_name=None,
                 max_sentences=None, name_filter=None):
        """Use 'condition' to filter names, e.g. 'Wikipedia'.
        """
        self.dir = dir_name
        self.lang = lang
        self.corpus_name = corpus_name
        self.max_sentences = max_sentences
        self.name_filter = name_filter
        self.index2word = None
        self.word2index = None
        self.tokenizer_filter = '"\'#$%&()*+/:<=>@[\\]^_`{|}~\t\n»«„“‹›'

    def build_word_index(self):
        """Create indices that map words to numbers and the other way around.
        """
        line_words = self.sentences(words=True)
        self.index2word = [ word for word, _ in Counter(chain.from_iterable(line_words)).most_common()]
        self.word2index = {word: index for index, word in enumerate(self.index2word)}

    def delete_word_index(self):
        """Delete word indices built with `build_word_index()`.
        """
        self.word2index = None
        self.index2word = None

    def preprocess_and_store(self, directory, log=None):
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.store_tokens(os.path.join(directory, 'tokens.txt'), log=log)
        self.store_numbers(os.path.join(directory, 'numbers.txt'), log=log)
        self.store_index2word(os.path.join(directory, 'index2word.txt'))

    def store_tokens(self, filename, log=None):
        with open(filename, 'w') as f:
            for i, sentence in enumerate(self.sentences(words=True)):
                if log and i % log == 0:
                    print("tokens - wrote %d lines" % i)
                f.write(" ".join(sentence) + '\n')

    def store_numbers(self, filename, log=None):
        with open(filename, 'w') as f:
            for i, numbers in enumerate(self.number_sentences()):
                if log and i % log == 0:
                    print("numbers - wrote %d lines" % i)
                f.write(" ".join(map(str, numbers)) + '\n')

    def store_triples(self, filename, log=None):
        with open(filename, 'w') as f:
            for i, ngram in self.number_ngrams:
                if log and i % log == 0:
                    print("triples - wrote %d lines" % i)
                f.write(" ".join(map(str, ngram)) + "\n")

    def store_index2word(self, filename):
        if not self.word2index:
            self.build_word_index()
        with open(filename, 'w') as f:
            for word in self.index2word:
                f.write(word + '\n')

    def number_sentences(self):
        if not self.word2index:
            self.build_word_index()
        for sentence in self.sentences(words=True):
            yield [self.word2index[word] for word in sentence]

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def number_ngrams(self, n):
        for numbers in self.number_sentences():
            for ngram in self.rolling_window(np.array(numbers), n):
                yield ngram

    def number_ngrams_(self, n):
        for numbers in self.number_sentences():
            for i in range(0, len(numbers) - n + 1):
                yield numbers[i:i+n]

    def sentences(self, words=True):
        """Iterate over lines in files returning sentences.
        """
        files = self.sentence_files()
        trans = str.maketrans('', '', self.tokenizer_filter)
        for f in files:
            for line in open(f):
                # Lines are of form: 'LineNumber\tActualSentence\n'
                sent = line.split('\t')[1].strip().lower().translate(trans)
                yield word_tokenize(sent) if words else sent

    def sentences_n(self, n):
        for i, s in enumerate(self.sentences()):
            if self.max_sentences and i >= n:
                break
            yield s

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
