# training steps for intermediate decoders

import torch

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon


def fit_probes_by_ridge_regression(model_wrapper, activity_dataset, inds,
                                   internal_dim=768, num_classes=1000):
    """ Use cross-validated ridge regression to fit linear probes to initialize
    each of the model classifiers
    Use these weights to initialize the model classifier.

    activity_dataset: dataset of internal activity created by generate_activity_dataset
    """
    activity_dataset.set_format('pt')

    layer_names = [model_wrapper.layers[i] for i in inds]
    probes = {}

    for layer in layer_names:
        probe = torch.nn.Linear(internal_dim, num_classes)

        train_activity = activity_dataset['train'][:][layer]
        train_labels = activity_dataset['train'][:]['label']

        # fit SVM model
        encoder = OneHotEncoder()
        targets = 2*encoder.fit_transform(train_labels.reshape(-1, 1)).toarray() - 1

        regressor = Ridge()
        best_reg_model = RandomizedSearchCV(
                                    estimator=regressor, 
                                    param_distributions={'alpha': expon(scale=100)},
                                    n_iter=15,
                                    cv=5, 
                                    scoring='neg_mean_squared_error',
                                    refit=True
                                  )

        best_reg_model.fit(train_activity, targets)

        weights = best_reg_model.best_estimator_.coef_
        bias = best_reg_model.best_estimator_.intercept_

        with torch.no_grad():
            probe.weight.copy_(torch.as_tensor(weights))
            probe.bias.copy_(torch.as_tensor(bias))

        probes[layer] = probe

    return probes
