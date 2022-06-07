from typing import List, Tuple, Optional

from tensorflow.python.keras.layers import (
    AveragePooling2D,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
    Concatenate,
    Flatten,
    Dense,
    Activation,
    Layer
)
from tensorflow.python.keras.models import Model

from constants.constants import (
    COMPONENT_WITHOUT_INPUT_GENE_SIZE,
    ReductionTypeEnum,
    CELL_PREFIX_SIZE,
    NETWORK_PREFIX_SIZE,
    NETWORK_FLATTEN_TYPES,
    NetworkFlattenType,
    ELIGIBLE_NETWORK_FLATTEN_TYPES, NETWORK_FLATTEN_TYPE_GENE_SIZE
)
from network_builder.translator.cell import Cell
from network_builder.translator.stem import Stem


class Network:
    def __init__(
            self,
            genome: str,
            n_cell: int,
            n_components_cell: int,
            n_base_filters: int,
            n_classes: int,
            component_gene_size: int = COMPONENT_WITHOUT_INPUT_GENE_SIZE
    ):
        self.n_cell: int = n_cell
        self.n_components_cell: int = n_components_cell
        self.n_base_filters: int = n_base_filters
        self.n_classes: int = n_classes
        self.component_gene_size: int = component_gene_size
        self.cell_list: List[Cell] = []
        self.stem: Optional[Stem] = None
        self.flatten_type: Optional[NetworkFlattenType] = None
        self.__translate_network_genome(genome)

    def print_debug(self):
        for cell in self.cell_list:
            print(cell)
            print("Components")
            for cmp in cell.cell_components:
                print(cmp)

    def __translate_network_stem(
            self,
            genome: str
    ):
        self.stem = Stem(
            genome,
            self.n_base_filters
        )

    def __translate_flatten_gene(
            self,
            genome: str
    ):
        print(f"ffff:{genome}")
        self.flatten_type = NETWORK_FLATTEN_TYPES[int(genome, 2) % ELIGIBLE_NETWORK_FLATTEN_TYPES]

    def __translate_network_genome(
            self,
            genome: str
    ) -> None:
        n_bits_comp: int = self.n_components_cell * self.component_gene_size
        n_bits_box: int = int(self.n_components_cell * (self.n_components_cell + 1) / 2)
        total_bits_per_cell: int = CELL_PREFIX_SIZE + n_bits_comp + n_bits_box
        self.__translate_network_stem(genome[:NETWORK_PREFIX_SIZE - NETWORK_FLATTEN_TYPE_GENE_SIZE])
        self.__translate_flatten_gene(genome[NETWORK_PREFIX_SIZE - NETWORK_FLATTEN_TYPE_GENE_SIZE:NETWORK_PREFIX_SIZE])
        cell_offset: int = NETWORK_PREFIX_SIZE
        n_filters: int = self.n_base_filters
        input_reduction_level: int = self.stem.output_reduction_level
        for i in range(self.n_cell):
            cell_genome: str = genome[cell_offset: cell_offset + i + total_bits_per_cell]
            cell_info = cell_genome[:CELL_PREFIX_SIZE + i]
            cell_cmp_genome = cell_genome[CELL_PREFIX_SIZE + i:]
            cell: Cell = Cell(
                cell_cmp_genome,
                cell_info,
                self.n_components_cell,
                n_filters,
                input_reduction_level,
                self.component_gene_size
            )
            if cell.reduction_type != ReductionTypeEnum.NO_REDUCTION:
                input_reduction_level *= 2

            self.cell_list.append(cell)
            cell_offset += total_bits_per_cell + i
            n_filters *= 2

    def __call__(
            self,
            input_layer: Layer
    ) -> Model:
        cell_output_list: List[Tuple[Layer, int]] = [(self.stem(input_layer), self.stem.output_reduction_level)]
        for cell in self.cell_list:
            cell_input_layers: List[Layer] = self.__fit_cell_input(
                cell,
                cell_output_list
            )
            cell_input: Layer = (
                Concatenate(axis=-1)(cell_input_layers) if len(cell_input_layers) > 1
                else cell_input_layers[0]
            )
            cell_output_list.append(
                (
                    cell(cell_input),
                    cell.output_reduction_level
                )
            )
        output_layer: Layer = cell_output_list[-1][0]
        output_layer = self.__build_flatten_layer(output_layer)
        output_layer = Dense(
            self.n_classes,
            activation="softmax"
        )(output_layer)

        model: Model = Model(
            inputs=input_layer,
            outputs=output_layer,
            name="RD1"
        )
        return model

    def __build_flatten_layer(
            self,
            output_layer: Layer
    ) -> Layer:
        if self.flatten_type == NetworkFlattenType.FLATTEN:
            return Flatten()(output_layer)
        elif self.flatten_type == NetworkFlattenType.GLOBAL_MAX_POOLING:
            return GlobalMaxPooling2D()(output_layer)
        elif self.flatten_type == NetworkFlattenType.GLOBAL_AVG_POOLING:
            return GlobalAveragePooling2D()(output_layer)

    def __fit_cell_input(
            self,
            cell: Cell,
            cell_output_list: List[Tuple[Layer, int]]
    ) -> List[Layer]:
        previous_cells: List[Layer] = []
        for i in range(len(cell.previous_cell_connections)):
            if cell.previous_cell_connections[i] == "1":
                previous_cells.append(
                    self.__fit_layer(
                        cell_output_list[i][1],
                        cell.input_reduction_level,
                        cell_output_list[i][0]
                    )
                )
        previous_cells.append(
            self.__fit_layer(
                cell_output_list[-1][1],
                cell.input_reduction_level,
                cell_output_list[-1][0]
            )
        )
        return previous_cells

    def __fit_layer(
            self,
            output_reduction_level: int,
            input_reduction_level: int,
            layer: Layer
    ) -> Layer:
        stride = input_reduction_level // output_reduction_level
        if stride == 1:
            return layer
        else:
            return AveragePooling2D(
                pool_size=1,
                strides=stride
            )(layer)
