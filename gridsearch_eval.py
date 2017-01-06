from goethe.evaluation import eval_models_from_gridsearch
import os
import json
import itertools as it
from gridsearch import model_config, sample_sizes

def model_name(sample_size, vec_size, epochs):
    return 'n{0}_size{1}_epochs{2}.model'.format(sample_size, vec_size, epochs)

if __name__ == '__main__':
    parameters = it.product(sample_sizes,
                            model_config['size'],
                            model_config['iter'])

    model_names = list([model_name(*params) for params in parameters])

    results = eval_models_from_gridsearch(model_names, 'models', 'evaluation/question-words.txt')
    for result, model_name in zip(results, model_names):
        with open(os.path.join('eval-results', model_name + '.json'), 'w+') as f:
            print('Evaluated model: ' + model_name)
            json.dump(result, f)
