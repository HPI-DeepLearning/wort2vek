# Goethe
Bringing word2vec to the German language.

## Getting started with the Leizpig Corpus

The [Leizpig Corpora Collection](http://corpora2.informatik.uni-leipzig.de/download.html) is a quick way to start training models for the German language. You can load a corpus and iterate its sentences with the following code:
```python
from goethe.corpora import LeipzigCorpus

sentences = LeipzigCorpus('path/containing/corpora')
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

## Model building

You can train models using [gensim](https://radimrehurek.com/gensim/):

```python
import gensim

sentences = LeipzigCorpus('path/containing/corpora')
model = gensim.models.Word2Vec(sentences)
```

## Evaluation

A trained model can be queried for semantic similarity. We can say e.g., "*Obama* to *USA* is what *Putin* is to *X*" and ask our model to return a word that matches *X*:

```python
>>> model.most_similar(['Putin', 'USA'], ['Obama'], topn=3)
[('Russland', 0.7132166028022766),
 ('USA,', 0.7057479619979858),
 ('China', 0.6795132160186768)]
```

To test our model on multiple such [queries](https://github.com/rshkv/goethe/blob/master/evaluation/bestmatch-questions.txt) you can use the `model_accuracy` function:

```python
>>> from goethe.evaluation import model_accuracy
>>> model_accuracy(model, 'evaluation/bestmatch-questions.txt', topn=5)
[('Land-Währung', 0.5238095238095238),
 ('Hauptstad-Land', 0.47619047619047616),
 ('Land-Kontinent', 0.34615384615384615),
 ('Land-Sprache', 0.15384615384615385),
 ('Politik', 0.0),
 ('Technik', 0.6666666666666666),
 ('Geschlecht', 0.5220588235294118)]
```

The resulting list contains a tuple for each section with its name and accuracy. The accuracy here is the percentage of 4-tuples in which the `topn` words returned by  `most_similar` contained the right word.

