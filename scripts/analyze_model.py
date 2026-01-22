# runs model analysis for a single model on prefetched dataset
import json
import pandas as pd

from setup import image_model_setup
from initialization.activity_dataset import generate_activity_dataset
from analysis import run_across_layers, apply_model_decoder, accuracy_random_CLS,\
                     linear_probe_by_ridge_regression

from training import fit_probes_by_ridge_regression


with open('metadata.json', 'r') as f:
    metadata = json.load(f)


def probe_characterization(model_name, dataset_name, num_classes,
                           run_model_decoder=False, device='mps'):
    """Characterize linear probes on model activity
    - model_name: huggingface model to analyze
    - dataset_name: prefetched local dataset
    - num_classes
    - run_model_decoder: make comparison to alternative decoding metrics
    """
    model, image_datasets, _ = image_model_setup(model_name, dataset_name, num_classes)

    splits = list(image_datasets.keys())

    # generated activity data
    activity_dataset = generate_activity_dataset(model, image_datasets,
                                                 splits=splits,
                                                 include_classifier_inputs=True,
                                                 output_dir='temp_activity_dataset',
                                                 device=device
                                                 )

    print('Fitting probes')
    model = fit_probes_by_ridge_regression(model, activity_dataset)

    # run the analyses
    all_results = []

    print('Starting analysis')
    probe_results = linear_probe_by_ridge_regression(activity_dataset, cvfold=5)
    all_results.append(probe_results)

    print('probe: ', probe_results)

    if run_model_decoder:
        decoder_results = apply_model_decoder(model, activity_dataset)
        all_results.append(decoder_results)

        print('decoder: ', decoder_results)

    randomization_results = run_across_layers(model, image_datasets,
                                              accuracy_random_CLS,  shuffle=None,
                                              device=device)
    all_results.append(randomization_results)

    print('randomization: ', randomization_results)

    results = pd.concat(all_results)

    results['model'] = model_name
    results['dataset'] = dataset_name

    results.to_csv('results.csv')
