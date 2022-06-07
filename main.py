from typing import List

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize

from data_aug.data_aug_processor import execute_augmentation_pipeline, get_final_folders, generate_data_iterators
from network_builder.network_problem import CNNProblem

import numpy as np


def main(
        input_dataset_folder: str = "./COVID-19_Radiography_Dataset",
        output_dataset_folder: str = "./processed_dataset",
        with_data_aug_pipeline: bool = False
) -> None:
    final_folders: List[str] = []
    if with_data_aug_pipeline:
        final_folders = execute_augmentation_pipeline(
            input_dataset_folder,
            output_dataset_folder
        )
    else:
        final_folders = get_final_folders(
            output_dataset_folder
        )

    batch_size: int = 32
    train_iterator, validation_iterator = generate_data_iterators(
        final_folders,
        batch_size=batch_size
    )

    my_problem = CNNProblem(
        n_components_box=5,
        n_boxes=4,
        input_size=(299, 299, 3),
        n_classes=4,
        n_base_filters=32,
        training_set_generator=train_iterator,
        validation_set_generator=validation_iterator,
        batch_size=batch_size,
        max_epochs=4
    )

    algorithm = NSGA2(
        pop_size=6,
        n_offsprings=10,
        sampling=get_sampling("bin_random"),
        crossover=get_crossover("bin_two_point"),
        mutation=get_mutation("bin_bitflip", prob=0.3),
        eliminate_duplicates=True
    )

    res = minimize(
        my_problem,
        algorithm,
        ('n_gen', 10),
        seed=1,
        verbose=True
    )
    print(res)


if __name__ == '__main__':
    main()
