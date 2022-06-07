from typing import List, Optional

from tensorflow.python.keras.layers import (
    MaxPooling2D,
    Conv2D,
    AveragePooling2D,
    Concatenate,
    Activation,
    BatchNormalization,
    Layer
)

from constants.constants import (
    ReductionTypeEnum,
    CellOutputTypeEnum, CELL_PREFIX_SIZE, REDUCTION_TYPES, REDUCTION_TYPE_GENE_SIZE, CELL_OUTPUT_TYPES
)
from network_builder.translator.cell_component import CellComponent


class Cell:
    def __init__(
            self,
            components_genome: str,
            cell_genome: str,
            n_components: int,
            n_filters: int,
            input_reduction_level: int,
            component_gene_size: int
    ):
        self.g1 = components_genome
        self.g2 = cell_genome
        self.cell_components: List[CellComponent] = []
        self.previous_cell_connections: List[str] = []
        self.reduction_type: Optional[ReductionTypeEnum] = None
        self.cell_output_type: Optional[CellOutputTypeEnum] = None
        self.input_reduction_level: int = input_reduction_level
        self.output_reduction_level: int = input_reduction_level
        self.n_components: int = n_components
        self.n_filters: int = n_filters
        self.component_gene_size: int = component_gene_size
        self.__translate_cell_info(cell_genome)
        self.__translate_cell_components(components_genome)
        if self.reduction_type != ReductionTypeEnum.NO_REDUCTION:
            self.output_reduction_level *= 2

    def __translate_cell_info(
            self,
            genome: str
    ) -> None:
        cell_prefix: str = genome[:CELL_PREFIX_SIZE]
        self.previous_cell_connections = list(genome[CELL_PREFIX_SIZE:])
        self.reduction_type = REDUCTION_TYPES[int(
            cell_prefix[:REDUCTION_TYPE_GENE_SIZE],
            2
        )]
        self.cell_output_type = CELL_OUTPUT_TYPES[int(
            cell_prefix[REDUCTION_TYPE_GENE_SIZE:],
            2
        )]

    def __translate_cell_components(
            self,
            genome: str
    ) -> None:
        comp_genome_size: int = self.n_components * self.component_gene_size
        comp_genome: str = genome[:comp_genome_size]
        con_genome: str = genome[comp_genome_size:]
        cmp_offset: int = 0
        con_offset: int = 0
        for i in range(self.n_components):
            cmp_g: str = comp_genome[cmp_offset: cmp_offset + self.component_gene_size]
            con_g: str = con_genome[con_offset: (con_offset + i + 1)]
            cell_component: CellComponent = CellComponent(
                cmp_g,
                con_g,
                self.n_filters
            )
            self.cell_components.append(cell_component)
            cmp_offset += self.component_gene_size
            con_offset += i + 1

    def __call__(self, input_layer: Layer) -> Layer:
        layer_list: List[Layer] = [input_layer]
        for cmp in self.cell_components:
            connected_layers: List[Layer] = self.__get_connected_layers(
                layer_list,
                cmp.connections
            )
            layer_list.append(cmp(connected_layers))

        bottom_layer_index: List[int] = self.__find_bottom_layers()
        bottom_layer_list: List[Layer] = []
        for layer_index in bottom_layer_index:
            bottom_layer_list.append(layer_list[layer_index + 1])

        return self.__build_output_layer(bottom_layer_list)

    def __build_output_layer(
            self,
            input_layer_list: List[Layer]
    ) -> Layer:
        # input_layer: Layer = (
        #     input_layer_list[0] if len(input_layer_list) == 1 else
        #     (
        #         Add()(input_layer_list) if self.cell_output_type == CellOutputTypeEnum.ADD else
        #         Concatenate()(input_layer_list)
        #     )
        # )
        input_layer: Layer = Concatenate(axis=-1)(input_layer_list)
        if self.reduction_type == ReductionTypeEnum.NO_REDUCTION:
            return input_layer
        elif self.reduction_type == ReductionTypeEnum.AVG_POOL:
            return AveragePooling2D(
                pool_size=3,
                strides=2,
                padding="same"
            )(input_layer)
        elif self.reduction_type == ReductionTypeEnum.MAX_POOL:
            return MaxPooling2D(
                pool_size=3,
                strides=2,
                padding="same"
            )(input_layer)

    def __get_connected_layers(
            self,
            layer_list: List[Layer],
            layer_connections: List[str]
    ) -> List[Layer]:
        connected_layers: List[Layer] = []
        if layer_connections[0] == "1":
            if layer_list[0].shape[-1] != self.n_filters:
                x = Conv2D(
                    filters=self.n_filters,
                    kernel_size=1,
                    strides=1,
                    kernel_initializer='he_normal',
                    padding="same",
                    use_bias=False
                )(layer_list[0])
                x = BatchNormalization(axis=-1)(x)
                x = Activation("relu")(x)
                connected_layers.append(x)
            else:
                connected_layers.append(layer_list[0])

        for i in range(1, len(layer_connections)):
            if layer_connections[i] == "1":
                connected_layers.append(layer_list[i])
        return connected_layers

    def __find_bottom_layers(
            self
    ) -> List[int]:
        bottom_layers: List[int] = []
        index_layer: int = 0
        for cmp in self.cell_components:
            if cmp.connections[0] == "1":
                bottom_layers += self.__find_bottom_layers_aux(index_layer + 1)
            index_layer += 1
        return list(dict.fromkeys(bottom_layers))

    def __find_bottom_layers_aux(
            self,
            layer_index: int
    ) -> List[int]:
        connected_layers_index: List[int] = []
        for i in range(layer_index, self.n_components):
            if self.cell_components[i].connections[layer_index] == "1":
                result: List[int] = self.__find_bottom_layers_aux(
                    i + 1
                )
                connected_layers_index += result
        return connected_layers_index if len(connected_layers_index) > 0 else [layer_index - 1]

    def __str__(self):
        return (f"cmp_g: {self.g1} \ncell_g: {self.g2}"
                f"\nFilters: {self.n_filters}\nReduction Type: {self.reduction_type}\n"
                f"Reduction I level: {self.input_reduction_level}\nReduction O level: {self.output_reduction_level}\n"
                f"Previous cell: {self.previous_cell_connections}\n")
