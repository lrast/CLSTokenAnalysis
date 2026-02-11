# Training script for probes
import sys

from src.model.setup import image_model_setup
from src.data.activity_dataset import OnlineLayerInputDataset
from src.model.CLS_token_probing import ModuleSpecificDecoder
from src.train.middle_decoders import train_module_decoder


def train_probes(mode, device='mps'):
    """ create the training probes  """
    model_name = "facebook/dinov2-base"
    dataset_name = "temp_dataset_subsample"

    # model for train data generate
    model, image_datasets, _ = image_model_setup(model_name, dataset_name, 1000)

    # second copy of the model: used for e
    model_analysis, _, _ = image_model_setup(model_name, dataset_name, 1000)
    model_analysis.to('mps')
    model_analysis.model.eval()

    layer_inds = [10, 8, 9, 11]

    for i in layer_inds:
        layer_name = f'dinov2.encoder.layer.{i}'

        ds_train = OnlineLayerInputDataset(model, layer_name, image_datasets['train'],
                                           device=device)
        ds_validation = OnlineLayerInputDataset(model, layer_name, image_datasets['validation'],
                                                device=device)

        if i == 10:
            probe = ModuleSpecificDecoder(mode=mode)
        else:
            probe = ModuleSpecificDecoder.from_pretrained(f'outs_{mode}/layer10_probe_{mode}',
                                                          mode=mode)

        base_module = model_analysis.module_dict[layer_name]

        model_out = train_module_decoder(probe, base_module, ds_train, ds_validation)
        model_out.save_pretrained(f'outs_{mode}/layer{i}_probe_{mode}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Usage: python probe_train.py <mode>")
        sys.exit(1)
    train_probes(mode)
