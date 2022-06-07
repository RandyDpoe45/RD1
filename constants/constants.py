from enum import Enum

COMPONENT_TYPE_GENE_SIZE = 3
COMPONENT_WITHOUT_INPUT_GENE_SIZE = 3
COMPONENT_WITH_INPUT_GENE_SIZE = 4
CELL_PREFIX_SIZE = 3
NETWORK_PREFIX_SIZE = 5
NETWORK_FLATTEN_TYPE_GENE_SIZE = 2
REDUCTION_TYPE_GENE_SIZE = 2
ELIGIBLE_COMPONENTS = 8
ELIGIBLE_NETWORK_FLATTEN_TYPES = 2


class CellComponentTypeEnum(Enum):
    IDENTITY = 0
    CONV_3x3 = 1
    CONV_5x5 = 2
    CONV_1x7_7x1 = 3
    MAX_POOL_3x3 = 4
    MAX_POOL_5x5 = 5
    AVG_POOL_3x3 = 6
    AVG_POOL_5x5 = 7


class CellComponentInputTypeEnum(Enum):
    ADD = 0
    CONCAT = 1


class ReductionTypeEnum(Enum):
    NO_REDUCTION = 0
    INTERNAL_REDUCTION = 1
    MAX_POOL = 2
    AVG_POOL = 3


class CellOutputTypeEnum(Enum):
    ADD = 0
    CONCAT = 1


class NetworkFlattenType(Enum):
    GLOBAL_AVG_POOLING = 0
    GLOBAL_MAX_POOLING = 1
    FLATTEN = 2


COMPONENT_INPUT_TYPE = {
    0: CellComponentInputTypeEnum.ADD,
    1: CellComponentInputTypeEnum.CONCAT
}

CELL_COMPONENTS = {
    0: CellComponentTypeEnum.IDENTITY,
    1: CellComponentTypeEnum.CONV_3x3,
    2: CellComponentTypeEnum.CONV_5x5,
    3: CellComponentTypeEnum.CONV_1x7_7x1,
    4: CellComponentTypeEnum.CONV_3x3,
    5: CellComponentTypeEnum.CONV_5x5,
    6: CellComponentTypeEnum.AVG_POOL_3x3,
    7: CellComponentTypeEnum.MAX_POOL_3x3,
}

REDUCTION_TYPES = {
    0: ReductionTypeEnum.NO_REDUCTION,
    1: ReductionTypeEnum.NO_REDUCTION,
    2: ReductionTypeEnum.AVG_POOL,
    3: ReductionTypeEnum.MAX_POOL
}

CELL_OUTPUT_TYPES = {
    0: CellOutputTypeEnum.ADD,
    1: CellOutputTypeEnum.CONCAT
}

NETWORK_FLATTEN_TYPES = {
    0: NetworkFlattenType.GLOBAL_AVG_POOLING,
    1: NetworkFlattenType.GLOBAL_MAX_POOLING,
    2: NetworkFlattenType.FLATTEN
}
