import os
from typing import List, Tuple

import numpy as np
import shutil
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def remove_destination_dataset_folders(
        dst_folder: str
) -> None:
    if os.path.exists(f"{dst_folder}/dataset_split"):
        shutil.rmtree(f"{dst_folder}/dataset_split", ignore_errors=True)

    if os.path.exists(f"{dst_folder}/dataset_aug"):
        shutil.rmtree(f"{dst_folder}/dataset_aug", ignore_errors=True)


def divide_dataset(
        input_folder: str,
        output_folder: str
) -> List[str]:
    input_dir: str = input_folder
    output_dir_aux: str = f"{output_folder}/dataset_split"
    aux_folders = [
        f"{output_dir_aux}/training_set",
        f"{output_dir_aux}/validation_set",
        f"{output_dir_aux}/test_set"
    ]

    os.makedirs(aux_folders[0], exist_ok=True)
    os.makedirs(aux_folders[1], exist_ok=True)
    os.makedirs(aux_folders[2], exist_ok=True)

    for class_lable in next(os.walk(input_dir))[1]:
        print(f"class {class_lable}")
        filename_array = np.array(os.listdir(f"{input_dir}/{class_lable}/images"))
        print(filename_array)
        np.random.shuffle(filename_array)
        train, validate, test = np.split(filename_array,
                                         [int(filename_array.shape[0] * 0.7), int(filename_array.shape[0] * 0.90)])
        for filename_list, directory in zip([train, validate, test], aux_folders):
            os.makedirs(f"{directory}/{class_lable}/", exist_ok=True)
            for filename in filename_list:
                shutil.copy(f"{input_dir}/{class_lable}/images/{filename}", f"{directory}/{class_lable}/{filename}")

    return aux_folders


def augment_data(
        output_folder: str,
        aux_folders: List[str],
        samples_multiplier: int = 2
) -> List[str]:
    def add_noise(img):
        '''Add random noise to an image'''
        VARIABILITY = 2
        deviation = VARIABILITY * np.random.random()
        noise = np.random.normal(0, deviation, img.shape)
        np.add(img, noise, out=img, casting="unsafe")
        np.clip(img, 0., 255.)
        return img

    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1. / 255,
        #     preprocessing_function=add_noise
    )

    test_validation_datagen = ImageDataGenerator(rescale=1. / 255)

    output_dir: str = f"{output_folder}/dataset_aug"
    final_folders = [
        f"{output_dir}/training_set",
        f"{output_dir}/validation_set",
        f"{output_dir}/test_set"
    ]

    os.makedirs(final_folders[0], exist_ok=True)
    os.makedirs(final_folders[1], exist_ok=True)
    os.makedirs(final_folders[2], exist_ok=True)

    for class_lable in next(os.walk(aux_folders[0]))[1]:
        print(f'{aux_folders[0]}/{class_lable}')
        dst_name = f'{final_folders[0]}/{class_lable}'
        os.makedirs(dst_name, exist_ok=True)
        gen = train_datagen.flow_from_directory(
            f'{aux_folders[0]}',
            target_size=(299, 299),
            batch_size=32,
            class_mode='categorical',
            classes=[class_lable],
            save_to_dir=dst_name,
            save_prefix='',
            save_format='png'
        )
        for i in range(gen.samples * samples_multiplier // 32):
            next(gen)

    for origin, dst in zip(aux_folders[1:], final_folders[1:]):
        for class_lable in next(os.walk(origin))[1]:
            print(f'{dst}/{class_lable}')
            dst_name = f'{dst}/{class_lable}'
            os.makedirs(dst_name, exist_ok=True)
            gen = test_validation_datagen.flow_from_directory(
                f'{origin}',
                target_size=(299, 299),
                batch_size=32,
                class_mode='categorical',
                classes=[class_lable],
                save_to_dir=dst_name,
                save_prefix='',
                save_format='png'
            )
            for i in range(gen.samples * samples_multiplier // 32):
                next(gen)

    return final_folders


def get_final_folders(
        output_folder: str
) -> List[str]:
    output_dir: str = f"{output_folder}/dataset_aug"
    final_folders = [
        f"{output_dir}/training_set",
        f"{output_dir}/validation_set",
        f"{output_dir}/test_set"
    ]
    return final_folders


def execute_augmentation_pipeline(
        input_folder: str,
        output_folder: str
) -> List[str]:
    remove_destination_dataset_folders(output_folder)
    aux_folders: List[str] = divide_dataset(input_folder, output_folder)
    final_folders: List[str] = augment_data(output_folder, aux_folders)
    return final_folders


def generate_data_iterators(
        final_folders: List[str],
        batch_size: int = 32
) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    image_gen = ImageDataGenerator()
    train_generator = image_gen.flow_from_directory(
        final_folders[0],
        target_size=(299, 299),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = image_gen.flow_from_directory(
        final_folders[1],
        target_size=(299, 299),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator

