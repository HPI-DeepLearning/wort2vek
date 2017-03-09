import os
import time
from collections import OrderedDict

import goethe

MODELS_PATH = 'models'
DATA_PATH = 'data/leipzig'

# Size
sizeconf = OrderedDict([('sg', [0, 1]),
                        ('size', [100, 300, 400, 600]),
                        ('iter', [5])])
sizelimits = [5_000_000]

# Iterations
iterconf = OrderedDict([('sg', [0, 1]),
                        ('size', [300]),
                        ('iter', [1, 5, 10, 20])])
iterlimits = [5_000_000]

# N
nconf = OrderedDict([('sg', [0, 1]),
                     ('size', [300]),
                     ('iter', [5])])
nlimits = [1_000_000, 5_000_000, 10_000_000, 20_000_000]

# Window
windowconf = OrderedDict([('sg', [0, 1]),
                     ('window', [5, 10, 20]),
                     ('iter', [5])])
windowlimits = [1_000_000]

#
nconf = OrderedDict([('sg', [0, 1]),
                     ('size', [300]),
                     ('iter', [5])])
nlimits = [1_000_000, 5_000_000, 10_000_000, 20_000_000]

configurations = [('Window', windowconf, windowlimits)]

if __name__ == '__main__':
    for confname, conf, limits in configurations:
        models = goethe.eval.gridsearch(DATA_PATH, conf, limits, MODELS_PATH)
        print(time.strftime(f'\n%Y-%m-%d %H:%M\t{confname.upper()}'))
        for model, name in models:
            model.save(os.path.join(MODELS_PATH, name))
            print(f"{time.strftime('%Y-%m-%d %H:%M')}\t{name}")
