import argparse
from collections import defaultdict
import gensim.models.keyedvectors
import pandas as pd


def accuracy(questions, model):
    """Takes paths to questions and model file.
    Returns the gensim's evaluation.
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(model)
    return model.wv.accuracy(questions)


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
                'incorrect': incorrect, 'total': total}

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
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate model and generate CSV with results')
    parser.add_argument('questions_file',
                        help='questions file in word2vec format')
    parser.add_argument('model_files', nargs='+',
                        help='one or more models to be evaluated')
    parser.add_argument('-c', '--counts',
                        help='create an output file for each model containing the counts of correct/incorrect/total answers')
    args = parser.parse_args()
    print(args)

    exit()
    questions_file, model_files = args.questions_file, args.model_files
    evaluate_dfs = [accuracy_as_df(accuracy(questions_file, m))
                    for m in model_files]

