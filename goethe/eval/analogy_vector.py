#! /usr/bin/Python

from gensim.models.keyedvectors import KeyedVectors
from scipy import spatial
from numpy import linalg
import argparse
import os

DEFAULT_OUTPUT_PATH = '/home/mst3/deeplearning/goethe/eval-results'

def output_category(count, sums):
	str = ''
	if count == 0: count = 1
	for i in range(0, len(sums)):
		sums[i] /= count
		str += " %.4f" % sums[i]
	#print(sums)
	return str

def calculate_sample(words, wvs):
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
	
	Cosine = 1-spatial.distance.cosine(diff1, diff2)
	Euclidean = 1-linalg.norm(diff1-diff2)/(linalg.norm(diff1)+linalg.norm(diff2))
	M_Cosine = 1-spatial.distance.cosine(miff1, miff2)
	M_Euclidean = 1-linalg.norm(miff1-miff2)/(linalg.norm(miff1)+linalg.norm(miff2))
	
	results = [Cosine, Euclidean, M_Cosine, M_Euclidean]
	return results
	
def calculate_detail(words, wvs):
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
	
	ta = 1-spatial.distance.cosine(m2+m3-m4, m1)
	tb = 1-spatial.distance.cosine(m4+m1-m3, m2)
	tc = 1-spatial.distance.cosine(m1+m4-m2, m3)
	td = 1-spatial.distance.cosine(m2+m3-m1, m4)
	
	Cosine = 1-spatial.distance.cosine(diff1, diff2)
	Euclidean = 1-linalg.norm(diff1-diff2)/(linalg.norm(diff1)+linalg.norm(diff2))
	M_Cosine = 1-spatial.distance.cosine(miff1, miff2)
	M_Euclidean = 1-linalg.norm(miff1-miff2)/(linalg.norm(miff1)+linalg.norm(miff2))
	
	results = [ta, tb, tc, td, Cosine, Euclidean, M_Cosine, M_Euclidean]
	return results

def evaluate_analogy_in_space(model, question, o_path, print_detail):
	print('Evaluating ' + model + ' with ' + question)
	wvs = KeyedVectors.load_word2vec_format(model, binary=True)
	print(model + ' loading finished.')
	
	class_count = 0
	class_oov_count = 0
	class_sum = [0.0, 0.0, 0.0, 0.0]
	all_count = 0
	all_oov_count = 0
	all_sum = [0.0, 0.0, 0.0, 0.0]
	current_category = ''
	
	output_file = o_path + '/' + model.split('/')[len(model.split('/'))-1].split('.')[0] + '.space'
	print(output_file)
	with open(question) as f_in, open(output_file, 'w') as f_out:
		header = 'Category Done OOV Cosine Euclidean M-Cosine M-Eucliden' + "\n"
		f_out.write(header)
		for line in f_in:
			words = line.split()
			if len(words) != 4:
				if len(words) == 0: continue
				
				if ": " not in line: continue
				else:
					if all_count != 0:
						f_out.write(current_category + ' ' + str(class_count) + ' ' + str(class_oov_count) + output_category(class_count, class_sum) + "\n")
						class_count = 0
						class_oov_count = 0
						class_sum = [0.0, 0.0, 0.0, 0.0]
					#print('new catagory' + line)
					current_category = line.split(' ')[len(line.split(' '))-1]
					current_category = current_category[:-1]
			else:
				OOV = False
				for w in words:
					if w not in wvs.vocab:
						class_oov_count += 1
						all_oov_count += 1
						OOV = True
						break
				
				if OOV: continue
				
				results = calculate_sample(words, wvs)
				if len(results) != len(class_sum):
					class_oov_count += 1
					all_oov_count += 1
					continue
				
				all_count += 1
				class_count += 1
				for i in range(0, len(results)):
					class_sum[i] += results[i]
					all_sum[i] += results[i]
				
		f_out.write(current_category + ' ' + str(class_count) + ' ' + str(class_oov_count) + output_category(class_count, class_sum) + "\n")
		f_out.write('All ' + str(all_count) + ' ' + str(all_oov_count) + output_category(all_count, all_sum) + "\n")
		print('Analogy Vector Evaluation Done!')
		
	if print_detail:
		print('Printing Details...')
		output_file += '.detail'
		with open(question) as f_in, open(output_file, 'w') as f_out:
			header = 'A B C D oov toA toB toC toD Cos Euc M-Cos M-Euc Prediction' + "\n"
			f_out.write(header)
			for line in f_in:
				words = line.split()
				if len(words) != 4:
					f_out.write("\n")
					continue
				else:
					OOV = False
					for w in words:
						if w not in wvs.vocab:
							OOV = True
							f_out.write(words[0] + ' ' + words[1] + ' ' + words[2] + ' ' + words[3] + ' ' + w + "\n")
							break				
					if OOV: continue
					
					details = words[0] + ' ' + words[1] + ' ' + words[2] + ' ' + words[3] + ' - '
					details += output_category(1, calculate_detail(words, wvs))
					pred = wvs.most_similar(positive=[words[2], words[1]], negative=[words[0]], topn=1)
					#print(pred[0][0] + ' ' + words[3])
					if words[3] == pred[0][0]:
						details += ' Hit'
					else:
						details += ' Miss ' + " %.4f" % pred[0][1]
					f_out.write(details + "\n")	
					

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description='Evaluate model and generate CSV with results')
	parser.add_argument('-m', '--model', required=True,
						help='the model to be evaluated, in .bin format')
	parser.add_argument('-q', '--questions', nargs='+', required=True,
						help='questions file in word2vec format')
	parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_PATH,
						help='folder to write output files')
	parser.add_argument('-d', '--detail', default=False,
                        help='whether to output item-level detail, default FALSE')
	args = parser.parse_args()
	
	for q in args.questions:
		evaluate_analogy_in_space(args.model, q, args.output, args.detail)