import argparse
import pandas as pd
import os
from .similarity import eval_similarity
from .analogy import multiple_accuracy_df, combine_accuracy_dfs


q_path = 'evaluation/analogy'
q_files = ['sem-question-words.txt', 'syn-question-words.txt']
QUESTION_FILES = [os.path.join(q_path, f) for f in q_files]

def evaluate(models):

    # analogy evaluation
    model_dfs = [multiple_accuracy_df(m, QUESTION_FILES) for m in models]
    analogy_df = combine_accuracy_dfs(model_dfs).T.round(3) * 100

    # similarity evaluation
    similarity_results = eval_similarity(models)
    similarity_dfs = []
    for name, df in similarity_results:
        df['model'] = name
        similarity_dfs.append(df)
    similarity_df = pd.concat(similarity_dfs)

    return analogy_df, similarity_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    print(type(args.models))
    anal_df, sim_df = evaluate(list(args.models))
    if args.output:
        anal_df.to_csv(os.path(args.output, 'analogy.csv'))
        sim_df.to_csv(os.path(args.output, 'similarity.csv'))
    else:
        print(anal_df)
        print(sim_df)
