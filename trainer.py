import os.path
from typing import List

import torch
from PIL import Image
from super_image import TrainingArguments, Trainer
from super_image.data import EvalMetrics, EvalDataset
from super_image.models.edsr.configuration_edsr import EdsrConfig
from super_image.models.edsr.modeling_edsr import EdsrModel
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from dataset import DATASET_NAME, AVAILABLE_SUBSETS, eval_dataset_pair
from models import AVAILABLE_MODELS


def create_training_args(model_name: str, subset_name: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=f'./results_{model_name}_{subset_name}',
        num_train_epochs=5,
    )


def benchmark_mixed_dataset():
    datasets = []
    for subset in AVAILABLE_SUBSETS[DATASET_NAME]:
        train_dataset, _ = eval_dataset_pair(DATASET_NAME, subset)
        datasets.append(train_dataset)

    final_dataset = ConcatDataset(datasets)
    trained_models = []
    _, eval_dataset = eval_dataset_pair(DATASET_NAME, "realistic_wild_x4")

    for model in AVAILABLE_MODELS:
        model_config = AVAILABLE_MODELS[model]

        training_model = model(model_config)
        training_args = create_training_args(model.__name__, "mixed")

        trainer = Trainer(
            model=training_model,
            args=training_args,
            train_dataset=final_dataset
        )

        trainer.train()
        trained_models.append(training_model)

    for subset in AVAILABLE_SUBSETS[DATASET_NAME]:
        _, eval_dataset = eval_dataset_pair(DATASET_NAME, subset)
        print_trained_models("mixed", eval_dataset, trained_models)


def benchmark_train_eval():
    for subset in AVAILABLE_SUBSETS[DATASET_NAME]:
        train_dataset, eval_dataset = eval_dataset_pair(DATASET_NAME, subset)

        trained_models = []
        for model in AVAILABLE_MODELS:
            model_config = AVAILABLE_MODELS[model]

            training_model = model(model_config)
            training_args = create_training_args(model.__name__, subset)

            trainer = Trainer(
                model=training_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )

            trainer.train()
            trained_models.append(training_model)

        print_trained_models(subset, eval_dataset, trained_models)


def eval_trained_models():
    path_edsr = [
        ('results_EdsrModel_bicubic_x4', 'bicubic_x4'),
        ('results_EdsrModel_realistic_difficult_x4', 'realistic_difficult_x4'),
        ('results_EdsrModel_realistic_wild_x4', 'realistic_wild_x4'),
    ]

    for pair in path_edsr:
        folder, subset = pair
        path = os.path.join(folder, 'pytorch_model_4x.pt')

        train_dataset, _ = eval_dataset_pair(DATASET_NAME, subset)

        model = EdsrModel(EdsrConfig(scale=4))
        model.load_state_dict(torch.load(path))
        model.eval()

        dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=False,
        )

        counter = 0

        for data in dataloader:
            inputs, _ = data
            predicted = model(inputs)

            save_tensor_as_png(inputs, os.path.join('div2k_edsr_test', subset, 'inputs', f'{counter}.png'))
            save_tensor_as_png(predicted, os.path.join('div2k_edsr_test', subset, 'predicted', f'{counter}.png'))

            counter += 1
            if counter == 3:
                break


def save_tensor_as_png(tensor, file_path):
    # Переводим тензор в формат (высота, ширина, каналы)
    image_data = tensor.detach().squeeze(0).permute(1, 2, 0).numpy()

    # Преобразовываем значения из диапазона [0, 1] в [0, 255]
    image_data = (image_data * 255).astype('uint8')

    # Создаем объект изображения PIL
    image = Image.fromarray(image_data)

    # Извлекаем путь к директории из файла
    directory = os.path.dirname(file_path)

    # Проверяем, существует ли директория, и создаем ее, если нет
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Сохраняем изображение в файл
    image.save(file_path, format='PNG')


def print_trained_models(subset: str,
                         eval_dataset: EvalDataset,
                         trained_models: List[nn.Module]) -> None:
    print(f"Subset - {subset}")
    for trained_model in trained_models:
        print(f"Model - {trained_model.__class__.__name__}")
        EvalMetrics().evaluate(trained_model, eval_dataset)
        print("==========================================================================")
