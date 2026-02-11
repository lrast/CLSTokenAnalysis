# Initialization script to download dataset subsets
# Functional, but not pretty
import json
import sys
import os
import shutil

from datasets import load_dataset, Dataset, Image
from pathlib import Path


def download_stratified_image_subsets(dataset_id, samples_per_class=250,
                                      splits=['train', 'validation'],
                                      split_mapping={'validation': 'test'},
                                      split_train_to_val=0.2,
                                      output_dir='temp_dataset_subsample',
                                      seed=42,
                                      in_RAM=True
                                      ):
    """Download and save a stratified sub-sampling of the dataset to produce
    class-balanced splits.

    Shuffles the dataset after downloading and before saving.
    """
    dataset_dict = load_dataset(dataset_id, streaming=True)

    with open('metadata.json', 'r') as f:
        metadata = json.load(f)

    num_classes = metadata['dataset'][dataset_id]["num_classes"]

    dataset_dict = dataset_dict.cast_column("image", Image(decode=False))

    def generate_samples(split=None):
        local_dataset = dataset_dict[split]

        remaining = {i: samples_per_class for i in range(num_classes)}
        total_to_find = num_classes * samples_per_class
        count_found = 0

        for batch in local_dataset:
            if count_found == total_to_find:
                break

            label = batch['label']

            if (count_found == 0) and (label not in remaining):
                # fail loundly if the first label is unknown
                raise ValueError('Unknown label value')

            if label in remaining:
                count_found += 1

                remaining[label] -= 1
                if remaining[label] == 0:
                    # remove it from the dict
                    del remaining[label]

                yield batch

    base_dir = Path(output_dir)
    for split in splits:
        # write the results to a dataset

        # handle dataset naming
        if split in split_mapping:
            split_name = split_mapping[split]
        else:
            split_name = split

        if not in_RAM:
            data_subset = Dataset.from_generator(generate_samples, keep_in_memory=True,
                                                 gen_kwargs={'split': split},
                                                 split=split_name)
        else:
            ram_cache_dir = "/dev/shm/hf_cache"
            os.makedirs(ram_cache_dir, exist_ok=True)
            data_subset = Dataset.from_generator(generate_samples, keep_in_memory=True,
                                                 gen_kwargs={'split': split},
                                                 split=split_name,
                                                 cache_dir=ram_cache_dir)

        data_subset = data_subset.shuffle(seed=seed)
        data_subset = data_subset.cast_column("image", Image(decode=True))
        data_subset = data_subset.class_encode_column("label")

        if split == 'train' and split_train_to_val is not None:
            split_subsets = data_subset.train_test_split(split_train_to_val,
                                                         seed=seed,
                                                         stratify_by_column="label")
            split_subsets['train'].to_parquet(base_dir / 'train/data.parquet')
            split_subsets['test'].to_parquet(base_dir / 'validation/data.parquet')

        else:
            data_subset.to_parquet(base_dir / split_name / 'data.parquet')

        if in_RAM:
            shutil.rmtree(ram_cache_dir)


# Running script. To do: more featured args parser
if __name__ == '__main__':
    model_name = sys.argv[1]
    download_stratified_image_subsets(model_name)
