import numpy as np
from six.moves.urllib.request import urlretrieve
import os

url = 'http://corpora2.informatik.uni-leipzig.de/downloads/'
path = 'data/'

base_news = ['deu_news_{}_{}-text.tar.gz', 'deu_news_{}_{}.tar.gz']
base_wiki = ['deu_wikipedia_{}_{}-text.tar.gz', 'deu_wikipedia_{}_{}.tar.gz']

year_start = 1995
year_end = 2015

years = range(year_start, year_end + 1)
sizes = ['3M', '1M', '300K', '100K', '30K', '10K']


def corpus_name(base, year, size):
    return base.format(year, size)


def download_url(corpus_name):
    return url + corpus_name


def corpus_path(corpus_name):
    return path + corpus_name


def try_download(corpus_name):
    store_path = corpus_path(corpus_name)
    if os.path.exists(store_path):
        return True, store_path
    try:
        url = download_url(corpus_name)
        filename, header = urlretrieve(url, store_path)
        return True, filename
    except:
        print("Can't download: " + corpus_name)
        return False, None


def download_corpora(base):
    for year in years:
        print('Year: ' + str(year))
        for size in sizes:
            print('Trying size: ' + size)
            name = corpus_name(base, year, size)
            successful, filename = try_download(name)
            if successful:
                print('Successful downloaded: ' + filename)
                break


def download_corpora_news():
    for base in base_news:
        download_corpora(base)


def download_corpora_wiki():
    for base in base_wiki:
        download_corpora(base)
