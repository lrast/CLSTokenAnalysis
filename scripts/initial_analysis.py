import hydra
import pandas as pd

from initialization import setup_model_ds_collator_images, setup_model_ds_collator_text
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
    "dataset_name": 'zh-plus/tiny-imagenet',
    "run_model_decoder": False,
    "output_name": 'results.csv'
}

cs = ConfigStore.instance()
cs.store(name="base_config", node=default_config)

# lookup table for the template and max_ind for different models
layer_lookup = {
    'google/vit-base-patch16-224': {'input_type': 'image',
                                    'template': 'vit.encoder.layer.{ind}',
                                    'max_ind': 12},
    'facebook/dinov2-base': {'input_type': 'image',
                             'template': 'dinov2.encoder.layer.{ind}',
                             'max_ind': 12},
    'openai/clip-vit-base-patch32': {'input_type': 'image',
                                     'template': 'vision_model.encoder.layers.{ind}',
                                     'max_ind': 12},
    'google-bert/bert-base-uncased': {'input_type': 'text',
                                      'template': 'model.bert.encoder.layer.{ind}',
                                      'max_ind': 12},
    'FacebookAI/roberta-large': {'input_type': 'text',
                                 'template': 'roberta.encoder.layer.{ind}', 
                                 'max_ind': 24},
    'microsoft/deberta-v3-base': {'input_type': 'text',
                                  'template': 'deberta.encoder.layer.{ind}',
                                  'max_ind': 12}
}

models = ['google/vit-base-patch16-224', 'facebook/dinov2-base', 'openai/clip-vit-base-patch32',
          'google-bert/bert-base-uncased', 'FacebookAI/roberta-large', 'microsoft/deberta-v3-base']
datasets = ['ILSVRC/imagenet-1k']


@hydra.main(version_base=None, config_name="base_config")
def probe_characterization(cfg: DictConfig):
    """Load, train probe, and run characterization of model"""

    input_type = layer_lookup[cfg.model_name]['input_type']
    template = layer_lookup[cfg.model_name]['template']
    max_ind = layer_lookup[cfg.model_name]['max_ind']

    if input_type == 'image':
        model, dataset, collator = setup_model_ds_collator_images(cfg.model_name,
                                                                  cfg.dataset_name)
    elif input_type == 'text':
        model, dataset, collator = setup_model_ds_collator_text(cfg.model_name,
                                                                cfg.dataset_name)
    else:
        raise ValueError('Unknown input type')

    # probe setup
    model = fit_probes_by_ridge_regression(model, dataset['train'], collator,
                                           num_train_points=200 * model.num_labels)

    all_results = []
    analyses = [accuracy_random_CLS, linear_probe_by_ridge_regression]
    if cfg.run_model_decoder:
        analyses.append(apply_model_decoder)

    for analysis in analyses:
        all_results.append(run_across_layers(model, dataset, analysis, template, max_ind))

    results = pd.concat(all_results)

    results['model'] = cfg.model_name
    results['dataset'] = cfg.dataset_name

    results.to_csv(Path(HydraConfig.get().runtime.output_dir) / cfg.output_name, index=False)


if __name__ == '__main__':
    probe_characterization()
