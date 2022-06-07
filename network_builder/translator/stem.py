from keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import (
    Conv2D,
    Activation,
    BatchNormalization,
    Layer
)

from constants.constants import (
    CellComponentTypeEnum
)


class Stem:

    def __init__(
            self,
            genome: str,
            base_filters: int
    ):
        self.component_type: CellComponentTypeEnum = (
            CellComponentTypeEnum.CONV_3x3 if genome[0] == "0"
            else CellComponentTypeEnum.CONV_5x5
        )
        self.n_filters: int = (
            base_filters if genome[1] == "0"
            else base_filters // 2
        )
        self.stride: int = 2
        self.output_reduction_level: int = (
            2 if genome[2] == "0"
            else 4
        )

    def __call__(self, input_layer: Layer) -> Layer:
        kernel_size: int = 3 if self.component_type == CellComponentTypeEnum.CONV_3x3 else 5
        x = Conv2D(
            filters=self.n_filters,
            kernel_size=kernel_size,
            strides=self.stride,
            kernel_initializer='he_normal',
            padding="same",
            use_bias=False
        )(input_layer)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        if self.output_reduction_level == 4:
            return MaxPooling2D(
                pool_size=3,
                strides=2,
                padding="same"
            )(x)

        return x
