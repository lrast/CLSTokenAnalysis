import hydra
import json
import pandas as pd

from initialization import image_model_setup, generate_activity_dataset
from training import fit_probes_by_ridge_regression
from analysis import run_across_layers, apply_model_decoder, accuracy_random_CLS,\
                     linear_probe_by_ridge_regression


from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pathlib import Path


# Starting with configurations stored locally
default_config = {
    "model_name": 'google/vit-base-patch16-224',
    "dataset_name": "ILSVRC/imagenet-1k",
    "run_model_decoder": False,
    "output_name": 'results.csv',
    "analysis_params": {"device": 'mps'},
    "seed": 42
}

cs = ConfigStore.instance()
cs.store(name="base_config", node=default_config)

with open('metadata.json', 'r') as f:
    metadata = json.load(f)


@hydra.main(version_base=None, config_name="base_config")
def probe_characterization(cfg: DictConfig):
    """Load, train probe, and run characterization of model"""

    model_cfg = metadata['model'][cfg.model_name]
    dataset_cfg = metadata['dataset'][cfg.dataset_name]

    num_classes = dataset_cfg['num_classes']

    match model_cfg['input_type']:
        case 'image':
            model, dataset, collator = \
                image_model_setup(cfg.model_name, cfg.dataset_name,
                                  num_classes=num_classes, samples_per_class=200,
                                  seed=cfg.seed, splits=dataset_cfg['splits']
                                  )

    layers_to_track = [model_cfg['template'].format(ind=i)
                       for i in range(model_cfg['max_ind'])]

    # standardized split names: the second split is used as the 'test' split
    if 'test' not in dataset_cfg['splits']:
        dataset['test'] = dataset.pop(dataset_cfg['splits'][1])

    # probe setup
    print('Fitting probes')
    model = fit_probes_by_ridge_regression(model, dataset['train'], collator,
                                           num_train_points=200*num_classes,
                                           shuffle=None, **cfg.analysis_params
                                           )


    print('Generating activity datasets')
    activity_ds = generate_activity_dataset(model, dataset, layers_to_track,
                                            200*num_classes,
                                            dataset_keys=['train', 'test'],
                                            shuffle=True, **cfg.analysis_params)


    # run the analyses
    all_results = []

    print('starting analysis')
    probe_results = linear_probe_by_ridge_regression(activity_ds, 200, cvfold=5)
    all_results.append(probe_results)

    if cfg.run_model_decoder:
        decoder_results = apply_model_decoder(model, activity_ds)
        all_results.append(decoder_results)

    randomization_results = run_across_layers(model, dataset, accuracy_random_CLS,
                                              layers_to_track, shuffle=None,
                                              **cfg.analysis_params)
    all_results.append(randomization_results)

    results = pd.concat(all_results)

    results['model'] = cfg.model_name
    results['dataset'] = cfg.dataset_name

    results.to_csv(Path(HydraConfig.get().runtime.output_dir) / cfg.output_name, index=False)


if __name__ == '__main__':
    probe_characterization()
