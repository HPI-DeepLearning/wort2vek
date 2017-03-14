# name =$2
# cv = "$name.cv.txt"
# wv = "$name.wv.txt"
# . / count_and_filter - train $1 - cvocab $cv - wvocab $wv - min - count 100
# . / word2vecf - train $1 - wvocab $wv - cvocab $cv - output "$name.vec" - dumpcv "$name.cv.vec" - size 300 - negative 15 - threads 10 - iters 10

# if __name__ == '__main__':
