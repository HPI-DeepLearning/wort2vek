import fasttext

parser = argparse.ArgumentParser(description='Script for training word vector models based on fasttext')
parser.add_argument('input_file', type=str, help='Input file with one sentence plain text per line')
parser.add_argument('target', type=str, help='target file name to store model in')
parser.add_argument('-s', '--size', type=int, default=300, help='dimension of word vectors')
parser.add_argument('-w', '--window', type=int, default=5, help='size of the sliding window')
parser.add_argument('-m', '--mincount', type=int, default=5, help='minimum number of occurences of a word to be considered')
parser.add_argument('-c', '--workers', type=int, default=4, help='number of worker threads to train the model')
parser.add_argument('-g', '--sg', type=int, default=1, help='training algorithm: Skip-Gram (1), otherwise CBOW (0)')
parser.add_argument('-i', '--hs', type=int, default=1, help='use of hierachical sampling for training')
parser.add_argument('-n', '--negative', type=int, default=0, help='use of negative sampling for training (usually between 5-20)')
parser.add_argument('-o', '--cbowmean', type=int, default=0, help='for CBOW training algorithm: use sum (0) or mean (1) to merge context vectors')
args = parser.parse_args()


parser.add_argument('--input_file', type=str, help='training file path (required)')
parser.add_argument('--output', type=str, help='output file path (required)')
parser.add_argument('--lr', type=float, help='learning rate [0.05]')
parser.add_argument('--lr_update_rate', type=int, help='change the rate of updates for the learning rate [100]')
parser.add_argument('--dim', type=int, help='size of word vectors [100]')
parser.add_argument('--ws',type=int, help='size of the context window [5]')
parser.add_argument('--epoch', type=int, help='number of epochs [5]')
parser.add_argument('--min_count', type=int, help='minimal number of word occurences [5]')
parser.add_argument('--neg', type=int, help='number of negatives sampled [5]')
parser.add_argument('--word_ngrams', type=int, help='max length of word ngram [1]')
parser.add_argument('--loss', type=str, choices=['ns', 'hs', 'softmax'], help='loss function {ns, hs, softmax} [ns]')
parser.add_argument('--bucket', type=int, help='number of buckets [2000000]')
parser.add_argument('--minn', type=int, help='min length of char ngram [3]')
parser.add_argument('--maxn', type=int, help='max length of char ngram [6]')
parser.add_argument('--thread', type=int, help='number of threads [12]')
parser.add_argument('--t', type=float, help='sampling threshold [0.0001]')
parser.add_argument('--silent', type=int, help='disable the log output from the C++ extension [1]')
parser.add_argument('--encoding', type=str, help='specify input_file encoding [utf-8]')
args = parser.parse_args()




if args.sg == 1:
    model = fasttext.skipgram(args.input_file, args.target)
elif args.sg == 0:
    model = fasttext.cbow(args.input_file, args.target)
else:
    raise Error
