#! /usr/bin/Python

from gensim.models.keyedvectors import KeyedVectors
from scipy import spatial
from numpy import linalg
import argparse
import sys

vector_file = sys.argv[1]
if len(sys.argv) != 6:
	print('arguments wrong!')
	print(len(sys.argv))
	exit()
else:
	words = [sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]]
	print(words)

	
wvs = KeyedVectors.load_word2vec_format(vector_file, binary=True)
print('WVs loaded.')
for w in words:
	if w not in wvs.vocab:
		print('out of vocab!')
		exit()

#print(wvs.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=3))

w1 = wvs[words[0]]
w2 = wvs[words[1]]
w3 = wvs[words[2]]
w4 = wvs[words[3]]

m1 = w1 / linalg.norm(w1)
m2 = w2 / linalg.norm(w2)
m3 = w3 / linalg.norm(w3)
m4 = w4 / linalg.norm(w4)

diff1 = w1 - w2
diff2 = w3 - w4
miff1 = m1 - m2
miff2 = m3 - m4

print('-------Word Space---------')
print('to word-4: ', 1-spatial.distance.cosine(m2+m3-m1, m4))
print('to word-3: ', 1-spatial.distance.cosine(m1+m4-m2, m3))
print('to word-2: ', 1-spatial.distance.cosine(m4+m1-m3, m2))
print('to word-1: ', 1-spatial.distance.cosine(m2+m3-m4, m1))
print('------Analogy Space-------')
print('     cosine: ', 1-spatial.distance.cosine(diff1, diff2))
print('  Euclidean: ', 1-linalg.norm(diff1-diff2)/(linalg.norm(diff1)+linalg.norm(diff2)))
print('   M-cosine: ', 1-spatial.distance.cosine(miff1, miff2))
print('M-Euclidean: ', 1-linalg.norm(miff1-miff2)/(linalg.norm(miff1)+linalg.norm(miff2)))


