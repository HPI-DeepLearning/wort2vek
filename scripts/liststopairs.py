"""
The list format is:
    word,   context1,   context2,   context3,   context4,   ...

It is possible that our lists contain spaces.
split() might therefore not have the expected result.
"""


import argparse
import itertools as it


def main(inputfile, outputfile):

    with open(inputfile) as inf, open(outputfile, 'w') as outf:

        for line in inf:
            line = line.rstrip('\n')
            tokens = line.split()

            # Check for '/SPACE'
            if len(' '.join(tokens)) < len(line):
                continue

            word, *context = tokens
            pairs = (f'{w} {c}\n' for w, c in zip(it.repeat(word), context))
            outf.writelines(pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pairs file from lists file')
    parser.add_argument('inputfile')
    parser.add_argument('outputfile')
    args = parser.parse_args()

    main(args.inputfile, args.outputfile)
