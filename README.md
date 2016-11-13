# Goethe
Bringing word2vec to the German language.

## Getting started with the Leizpig Corpus

The [Leizpig Corpora Collection](http://corpora2.informatik.uni-leipzig.de/download.html) is a quick way to start training models for the German language. You can load a corpus and iterate its sentences with the following code:
```python
from goethe.corpora import LeipzigCorpus

corpus = LeipzigCorpus('path/containing/corpora')
sentences = list(corpus)
```

Assuming that you have a file structure like this:
```
path/containing/corpora/
    deu_news_2015_3M/
        deu_news_2015_3M-sentences.txt
        ...
    deu_wikipedia_2014_3M/
        deu_wikipedia_2014_3M-sentences.txt
        ...
    ...
```
