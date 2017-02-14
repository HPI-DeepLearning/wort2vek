import os
import time

import goethe

MODELS_PATH = 'models'
DATA_PATH = 'data/leipzig'

configurations = {'size': [100, 300, 600],
                  'sg': [0, 1],
                  'iter': [5, 10, 20], }
sentence_limits = [1_000_000, 5_000_000, 10_000_000]

if __name__ == '__main__':
    models = goethe.eval.gridsearch(DATA_PATH, configurations, sentence_limits)
    print(time.strftime('%Y-%m-%d %H:%M'))
    for model, name in models:
        model.save(os.path.join(MODELS_PATH, name))
        print(f"{time.strftime('%Y-%m-%d %H:%M')}\t{name}")
