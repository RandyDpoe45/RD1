from typing import List

from tensorflow.python.keras.layers import (
    SeparableConvolution2D,
    MaxPooling2D,
    Conv2D,
    AveragePooling2D,
    BatchNormalization,
    Add,
    Concatenate,
    Lambda,
    Activation,
    Layer
)

from constants.constants import (
    CELL_COMPONENTS,
    COMPONENT_WITHOUT_INPUT_GENE_SIZE,
    CellComponentTypeEnum,
    COMPONENT_TYPE_GENE_SIZE, CellComponentInputTypeEnum, COMPONENT_INPUT_TYPE
)


class CellComponent:
    cell_index: int = 0

    def __init__(
            self,
            component_genome: str,
            connections_genome: str,
            n_filters: int
    ):
        self.n_filters: int = n_filters
        self.cell_index: int = CellComponent.cell_index
        CellComponent.cell_index += 1
        self.component_type: CellComponentTypeEnum = CELL_COMPONENTS[
            int(component_genome[:COMPONENT_TYPE_GENE_SIZE], 2)
        ]

        self.input_type: CellComponentInputTypeEnum = (
            CellComponentInputTypeEnum.ADD if len(component_genome) == COMPONENT_WITHOUT_INPUT_GENE_SIZE else
            COMPONENT_INPUT_TYPE[int(component_genome[COMPONENT_TYPE_GENE_SIZE], 2)]
        )

        self.connections: List[str] = list(connections_genome)
        if "1" not in self.connections:
            self.connections[0] = "1"

    def __str__(self):
        return f"Cmp: {self.component_type}, input_type: {self.input_type} \nconn: {self.connections}\nIndex:{self.cell_index}\n"

    def __call__(
            self,
            input_layer_list: List[Layer]
    ) -> Layer:
        x: Layer = (
            input_layer_list[0] if len(input_layer_list) == 1 else
            (
                Add()(input_layer_list) if self.input_type == CellComponentInputTypeEnum.ADD else
                Concatenate()(input_layer_list)
            )
        )
        x = self.__squeeze_channels(x)
        x = self.__build_layer(x)

        return x

    def __squeeze_channels(
            self,
            layer: Layer
    ) -> Layer:
        if layer.shape[-1] != self.n_filters:
            x = Conv2D(
                filters=self.n_filters,
                kernel_size=1,
                strides=1,
                kernel_initializer='he_normal',
                padding="same",
                use_bias=False
            )(layer)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
            return x
        return layer

    def __build_layer(
            self,
            input_layer: Layer
    ) -> Layer:
        if self.component_type == CellComponentTypeEnum.IDENTITY:
            return Lambda(
                lambda x: x,
                name=f"{self.cell_index}-Iden"
            )(input_layer)
        elif (
                self.component_type == CellComponentTypeEnum.CONV_3x3
                or self.component_type == CellComponentTypeEnum.CONV_5x5
        ):
            kernel_size: int = (3 if self.component_type == CellComponentTypeEnum.CONV_3x3 else 5)
            x = SeparableConvolution2D(
                self.n_filters,
                kernel_size=kernel_size,
                kernel_initializer='he_normal',
                strides=1,
                padding='same',
                use_bias=False,
                name=f"{self.cell_index}-conv_{kernel_size}x{kernel_size}"
            )(input_layer)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
            return x

        elif self.component_type == CellComponentTypeEnum.CONV_1x7_7x1:
            x = Conv2D(
                self.n_filters,
                kernel_size=(1, 7),
                kernel_initializer='he_normal',
                strides=1,
                padding='same',
                use_bias=False,
                name=f"{self.cell_index}-conv_1x7"
            )(input_layer)
            x = Activation("relu")(x)
            x = Conv2D(
                self.n_filters,
                kernel_size=(7, 1),
                kernel_initializer='he_normal',
                strides=1,
                padding='same',
                use_bias=False,
                name=f"{self.cell_index}-conv_7x1"
            )(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
            return x

        elif (
                self.component_type == CellComponentTypeEnum.AVG_POOL_3x3
                or self.component_type == CellComponentTypeEnum.AVG_POOL_5x5
        ):
            pool_size: int = (3 if self.component_type == CellComponentTypeEnum.AVG_POOL_3x3 else 5)
            x = AveragePooling2D(
                pool_size=pool_size,
                strides=1,
                padding="same",
                name=f"{self.cell_index}-avgP_{pool_size}x{pool_size}"
            )(input_layer)
            return x

        elif (
                self.component_type == CellComponentTypeEnum.MAX_POOL_3x3
                or self.component_type == CellComponentTypeEnum.MAX_POOL_5x5
        ):
            pool_size: int = (3 if self.component_type == CellComponentTypeEnum.MAX_POOL_3x3 else 5)
            x = MaxPooling2D(
                pool_size=pool_size,
                strides=1,
                padding="same",
                name=f"{self.cell_index}-maxP_{pool_size}x{pool_size}"
            )(input_layer)
            return x
