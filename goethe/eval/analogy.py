import os
import argparse
from collections import defaultdict
import gensim.models.keyedvectors
import pandas as pd

# Names used in output files
CORRECT = 'correct'  # correctly answered questions
INCORRECT = 'incorrect'  # incorrectly answered questions
ANSWERED = 'answered'  # answered questions in total (correct + incorrect)
SECTION = 'section'  # name of sections in questions file (start with ':')
ACCURACY = 'accuracy'  # name accuracy
TOTAL = '(total)'  # identifier for summary sections
QUESTIONS = 'questions'  # number of questions in a section
OOV = 'oov'  # out of vocabulary rate
MODEL = 'model'  # name of models
OUT_EXT = '.analogy'  # extension used for output files


def name(path):
    """Strip path and extension from name.
    """
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def end_with_totals(df):
    """Move rows with TOTAL in the section name to the end.
    """
    non_totals = [i for i in df.index if TOTAL not in i]
    totals = [i for i in df.index if TOTAL in i]
    return df.reindex(non_totals + totals)


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
    counts = pd.Series(counts).astype(int)
    counts.name = name(questions)
    return counts


def accuracy_df(model, questions):
    """Create a dataframe with every section name as row and evaluation results
    (accuracy, correct/incorrect/total answers) as columns.
    """
    sections = (gensim.models.KeyedVectors
                .load_word2vec_format(model)
                .accuracy(questions))

    def collapse_section(s):
        """Take section and return results as dictionary."""
        section_name = s['section']
        correct = len(s['correct'])
        incorrect = len(s['incorrect'])
        answered = correct + incorrect
        accuracy = correct / answered if answered else 0
        return {SECTION: section_name, ACCURACY: accuracy, CORRECT: correct,
                INCORRECT: incorrect, ANSWERED: answered}

    collapsed_sections = [collapse_section(s) for s in sections]
    return (pd.DataFrame.from_records(collapsed_sections, index=SECTION)
            .assign(**{QUESTIONS: count_sections(questions)})
            .assign(**{OOV: lambda df: 1 - (df[ANSWERED] / df[QUESTIONS])})
            .rename({'total': f'{name(questions)} {TOTAL}'}))


def multiple_accuracy_df(model, questions):
    """Create accuracy dataframe for single model with multiple question files.
    """
    df = pd.concat(accuracy_df(model, q) for q in questions)
    # Divide sum by two because of the per file total columns
    total = df.sum().div(2).rename(TOTAL).astype(int)
    df = df.append(total)
    df.loc[TOTAL, ACCURACY] = (df.loc[TOTAL, CORRECT]
                               / df.loc[TOTAL, ANSWERED])
    df.name = name(model)
    return df


def combine_accuracy_dfs(model_dfs):
    """Combine accuracies of multiple dataframes into single dataframe of accuracies
    per model per question file.
    """
    combined_df = pd.concat([df[ACCURACY] for df in model_dfs], axis=1)
    combined_df.columns = [df.name for df in model_dfs]
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate model and generate CSV with results')
    parser.add_argument('-q', '--questions', nargs='+', required=True,
                        help='questions file in word2vec format')
    parser.add_argument('-m', '--models', nargs='+', required=True,
                        help='one or more models to be evaluated')
    parser.add_argument(
        '-o', '--output', help='folder to write output files, omission will cause printing')
    parser.add_argument('-c', '--counts', default=False, action='store_true',
                        help='create an output file for each model containing the counts of correct/incorrect/total answers')
    args = parser.parse_args()

    model_dfs = [multiple_accuracy_df(m, args.questions)
                 for m in args.models]

    combined_df = combine_accuracy_dfs(model_dfs)
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        path = os.path.join(args.output, f'{ACCURACY}{OUT_EXT}')
        combined_df.round(3).to_csv(path, index_label=SECTION)
    else:
        print(combined_df)

    if args.counts:
        for df in model_dfs:
            if args.output:
                df.round(3).to_csv(os.path.join(args.output, f'{df.name}{OUT_EXT}'),
                                   index_label=SECTION)
            else:
                print(df)
