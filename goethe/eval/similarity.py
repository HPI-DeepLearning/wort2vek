from gensim.models.keyedvectors import KeyedVectors
import os
import pandas as pd
import argparse

header = ['testset', 'oov', 'pearson_corr', 'pearson_p', 'spearman_corr', 'spearman_p']

eval_path = 'evaluation/similarity'
pairs_65 = 'pairs65.txt'
pairs_222 = 'pairs222.txt'
pairs_350 = 'pairs350.txt'
pairs_filenames = [pairs_65, pairs_222, pairs_350]

def eval_similarity(model_filenames):
    for model_filename in model_filenames:
        model = KeyedVectors.load_word2vec_format(model_filename)
        rows = []
        for pairs_filename in pairs_filenames:
            pair_file = os.path.join(eval_path, pairs_filename)
            sims = model.evaluate_word_pairs(pair_file)
            testset_name = pairs_filename.split(os.path.extsep)[0]
            row = [testset_name, sims[2], *sims[0], *sims[1]]
            rows.append(row)
        model_name = os.path.basename(model_filename).split(os.path.extsep)[0]
        df = pd.DataFrame(rows, columns=header)
        yield model_name, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
       description='''Evaluate the Model using Pearson correlation coefficient
                      and Spearman rank-order correlation coefficient between
                      the similarities from the dataset and the similarities
                      produced by the model itself.''')
    parser.add_argument('model_filenames', nargs='+',
                       help='one or more models to be evaluated')
    parser.add_argument('--output', help='path in which the results will be stored')

    args = parser.parse_args()
    models = args.model_filenames
    output = args.output

    df_pairs = eval_similarity(models)

    if output:
        for name, df in df_pairs:
            df.to_csv(os.path.join(output, name + '.similarity.csv'))
    else:
        for name, df in df_pairs:
            print(name)
            print(df)
