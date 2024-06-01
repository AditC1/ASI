# Source: Deep Learning for the Life Sciences
import deepchem as dc
import numpy as np
x = np.random.random((4, 5))
y = np.random.random((4, 1))
x, y
dataset = dc.data.NumpyDataset(x, y)
print(dataset.X)
print(dataset.y)
np.array_equal(x, dataset.X)
np.array_equal(y, dataset.y)
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
tox21_tasks

'''Each of the 12 tasks here corresponds with a particular biological 
experiment. In this case, each of these tasks is for an enzymatic assay 
which measures whether the molecules in the Tox21 dataset bind with the 
biological target in question. The terms NR-AR and so on correspond with 
these targets. In this case, each of these targets is a particular enzyme 
believed to be linked to toxic responses to potential therapeutic molecules.'''
tox21_datasets
'''tuple containing multipledc.data.Dataset objects: correspond to 
the training, validation, and test sets'''
train_dataset, valid_dataset, test_dataset = tox21_datasets
train_dataset.X.shape
valid_dataset.X.shape
test_dataset.X.shape
transformers
model = dc.models.MultitaskClassifier(n_tasks=12,
    n_features=1024,
    layer_sizes=[1000])
model.fit(train_dataset, nb_epoch=10)
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)
print(train_scores)
print(test_scores)