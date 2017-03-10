import os
import argparse
from collections import defaultdict
import gensim.models.keyedvectors
import pandas as pd


def accuracy(questions, model):
    """Takes paths to questions and model file.
    Returns the gensim's evaluation.
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(model)
    return model.accuracy(questions)


def accuracy_as_df(sections):
    """Create a dataframe with every section name as row and evaluation results
    (accuracy, correct/incorrect/total answers) as columns.
    """

    def collapse_section(s):
        """Take section and return results as dictionary."""
        name = s['section']
        correct = len(s['correct'])
        incorrect = len(s['incorrect'])
        total = correct + incorrect
        accuracy = round(correct / total, 2) if total else 0
        return {'name': name, 'accuracy': accuracy, 'correct': correct,
                'incorrect': incorrect, 'answered': total}

    collapsed_sections = [collapse_section(s) for s in sections]
    return pd.DataFrame.from_records(collapsed_sections, index='name')


def count_sections(questions):
    """Count questions per section."""
    counts = defaultdict(int)
    current = None
    with open(questions) as q:
        for line in q:
            if line.startswith(':'):
                current = line.split(':')[1].strip()
            else:
                counts[current] += 1
    counts['total'] = sum(counts.values())
    return pd.Series(counts).astype(int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate model and generate CSV with results')
    parser.add_argument('questions_file',
                        help='questions file in word2vec format')
    parser.add_argument('model_files', nargs='+',
                        help='one or more models to be evaluated')
    parser.add_argument('-o', '--output', default='.',
                        help='folder to write output files')
    parser.add_argument('-c', '--counts', default=False, action='store_true',
                        help='create an output file for each model containing the counts of correct/incorrect/total answers')
    args = parser.parse_args()

    counts = count_sections(args.questions_file)
    dfs = []
    for m in args.model_files:
        results = accuracy(args.questions_file, m)
        df = accuracy_as_df(results)
        df = df.assign(total=counts)
        df.name = os.path.splitext(os.path.basename(m))[0]
        dfs.append(df)

    os.makedirs(args.output, exist_ok=True)
    if args.counts:
        for df in dfs:
            df.to_csv(os.path.join(args.output, 'analogy.' + df.name))
    else:
        accuracy_df = pd.concat([df['accuracy'] for df in dfs], axis=1).T
        accuracy_df.index = [df.name for df in dfs]
        accuracy_df.to_csv(os.path.join(args.output, 'analogy.accuracy'),
                           index_label='model')
