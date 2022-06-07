from typing import Tuple

from keras.utils.vis_utils import plot_model
from tensorflow.python.keras.layers import (
    Input
)
from tensorflow.python.keras.models import Model

from constants.constants import (
    COMPONENT_WITHOUT_INPUT_GENE_SIZE,
    CELL_PREFIX_SIZE, NETWORK_PREFIX_SIZE, COMPONENT_WITH_INPUT_GENE_SIZE
)
from data_aug.data_aug_processor import get_final_folders, generate_data_iterators
from network_builder.translator.network import Network


def translate_genome(
        genome: str,
        n_components_box: int,
        n_cells: int,
        input_shape: Tuple[int, int, int],
        n_init_filters: int,
        n_classes: int,
        component_gene_size: int = COMPONENT_WITH_INPUT_GENE_SIZE
) -> Model:
    network: Network = Network(
        genome,
        n_cells,
        n_components_box,
        n_init_filters,
        n_classes,
        component_gene_size=component_gene_size
    )
    network.print_debug()
    model: Model = network(Input(shape=input_shape))
    return model


def test_translation() -> None:
    n_comp_box: int = 5
    n_cells: int = 4
    #
    n_bits_comp: int = n_comp_box * COMPONENT_WITHOUT_INPUT_GENE_SIZE
    n_bits_box: int = int(n_comp_box * (n_comp_box + 1) / 2)

    total_bits_per_cell: int = CELL_PREFIX_SIZE + n_bits_comp + n_bits_box
    total_bits: int = NETWORK_PREFIX_SIZE
    for i in range(n_cells):
        total_bits += total_bits_per_cell + i

    bits = "1011011001000101001010100011111001001111100010010010100110000010100001011001000010011000110110100100001001101110001000101010110000001101101101110010010111101100001"
    print(f"{total_bits} - {len(bits)}")
    print(bits)
    final_folders = get_final_folders(
        "../processed_dataset"
    )
    batch_size: int = 32

    train_iterator, validation_iterator = generate_data_iterators(
        final_folders,
        batch_size=batch_size
    )

    model = translate_genome(bits, n_comp_box, n_cells, (299, 299, 3), 32, 4)
    model.summary()
    print(model.count_params())
    plot_model(
        model, to_file='../model_plots/test2.png', show_shapes=True, show_dtype=True,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
        layer_range=None, show_layer_activations=True
    )



if __name__ == '__main__':
    test_translation()
