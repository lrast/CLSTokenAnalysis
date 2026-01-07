# analysis utilities
import torch
from tqdm import tqdm


def model_accuracy(model, dataset, **kwargs):
    """ Want: shuffle, larger batch sizes so that shuffling tokens gives signal
    """
    defaults = {'batch_size': 128, 'shuffle': True}

    dl = torch.utils.data.DataLoader(dataset, (defaults | kwargs))
    outputs = []
    
    for batch in tqdm(iter(dl)):
        images, labels = batch
        preds = model.forward(images.to('mps')).logits.argmax(1).cpu()
    
        outputs.append(preds.cpu() == labels)

    return torch.concat(outputs)


