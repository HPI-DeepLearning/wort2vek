import sys
import spacy

LANG = 'de'
BATCH_SIZE = 10000
N_THREADS = 4

input_file = sys.argv[1]

nlp = spacy.load(LANG)

def word_ctx(word):
    return (str(word.head) + " " + str(word.dep_) + "_" + str(word)).lower()

def inv_word_ctx(word):
    return (str(word) + " " + str(word.dep_) + "I" + "_" + str(word.head)).lower()

def doc2ctxpairs(doc):
    for word in doc:
        if not word == word.head:
            print(word_ctx(word))
            print(inv_word_ctx(word))

with open(input_file, 'r') as f:
    lines = (line.strip() for line in f)
    docs = nlp.pipe(lines, batch_size=BATCH_SIZE, n_threads=N_THREADS)
    for doc in docs:
        doc2ctxpairs(doc)
