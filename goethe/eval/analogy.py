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


def accuracy_df(questions, model):
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
            .assign(questions=count_sections(questions))
            .rename({'total': f'{name(questions)} {TOTAL}'}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate model and generate CSV with results')
    parser.add_argument('-q', '--questions', nargs='+',
                        help='questions file in word2vec format')
    parser.add_argument('-m', '--models', nargs='+',
                        help='one or more models to be evaluated')
    parser.add_argument('-o', '--output', default='.',
                        help='folder to write output files')
    parser.add_argument('-c', '--counts', default=False, action='store_true',
                        help='create an output file for each model containing the counts of correct/incorrect/total answers')
    parser.add_argument('-t', '--totals', default=False, action='store_true',
                        help=f"move all summary sections (containing '{TOTAL}') to the end")
    parser.add_argument('-p', '--print', default=False, action='store_true',
                        help='print output instead of writing to file')
    args = parser.parse_args()

    model_dfs = []
    for m in args.models:
        df = pd.concat(accuracy_df(q, m) for q in args.questions)
        # Divide sum by two because of the per file total columns
        total = df.sum().div(2).rename(TOTAL).astype(int)
        df = df.append(total)
        df.loc[TOTAL, ACCURACY] = (df.loc[TOTAL, CORRECT]
                                   / df.loc[TOTAL, ANSWERED])
        if args.totals:
            df = end_with_totals(df)
        df.name = name(m)
        model_dfs.append(df)

    os.makedirs(args.output, exist_ok=True)
    if args.counts:
        for df in model_dfs:
            if args.print:
                print(df)
            else:
                df.to_csv(os.path.join(args.output, f'{df.name}{OUT_EXT}'),
                          index_label=SECTION)
    else:
        accuracy_df = pd.concat([df[ACCURACY] for df in model_dfs], axis=1)
        accuracy_df.columns = [df.name for df in model_dfs]
        if args.print:
            print(accuracy_df)
        else:
            path = os.path.join(args.output, f'{ACCURACY}{OUT_EXT}')
            accuracy_df.to_csv(index_label=MODEL, path=path)
