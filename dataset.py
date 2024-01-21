from typing import Tuple

from datasets import load_dataset
from super_image.data import TrainDataset, EvalDataset, augment_five_crop

DATASET_NAME = 'eugenesiow/Div2k'
AVAILABLE_SUBSETS = {
    'eugenesiow/Div2k': ['bicubic_x4', 'realistic_wild_x4', 'realistic_difficult_x4']
}


def eval_dataset_pair(name: str, degradation_type: str) -> Tuple[TrainDataset, EvalDataset]:
    augmented_dataset = load_dataset(name, degradation_type, split='train') \
        .map(augment_five_crop, batched=True, desc="Augmenting Dataset", num_proc=28)
    train_dataset = TrainDataset(augmented_dataset)
    eval_dataset = EvalDataset(load_dataset(name, degradation_type, split='validation'))

    return train_dataset, eval_dataset
