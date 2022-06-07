from typing import Tuple, List

import numpy as np
from pymoo.core.problem import Problem
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.metrics import Precision, Recall, AUC

from constants.constants import COMPONENT_WITHOUT_INPUT_GENE_SIZE, CELL_PREFIX_SIZE, NETWORK_PREFIX_SIZE
from network_builder.network_translator import translate_genome


class CNNProblem(Problem):
    def __init__(
            self,
            n_components_box: int,
            n_boxes: int,
            input_size: Tuple[int, int, int],
            n_classes: int,
            n_base_filters: int,
            training_set_generator,
            validation_set_generator,
            batch_size: int,
            max_epochs: int = 4
    ):
        n_bits_comp: int = n_components_box * COMPONENT_WITHOUT_INPUT_GENE_SIZE
        n_bits_box: int = int(n_components_box * (n_components_box + 1) / 2)

        total_bits_per_cell: int = CELL_PREFIX_SIZE + n_bits_comp + n_bits_box
        total_bits: int = NETWORK_PREFIX_SIZE
        for i in range(n_boxes):
            total_bits += total_bits_per_cell + i

        self.n_components_box: int = n_components_box
        self.n_boxes: int = n_boxes
        self.input_size: Tuple[int, int, int] = input_size
        self.training_set_generator = training_set_generator
        self.validation_set_generator = validation_set_generator
        self.max_epochs: int = max_epochs
        self.n_classes: int = n_classes
        self.n_base_filters: int = n_base_filters
        self.batch_size = batch_size
        self.log_list = []

        super().__init__(
            n_var=total_bits,
            n_obj=2,
            n_constr=1,
            xl=0,
            xu=1
        )

    @staticmethod
    def get_total_params(model: Model) -> int:
        return model.count_params()

    def save_logs(self, log_line) -> None:
        with open("./log_file.txt", "a") as f:
            f.write(log_line)
            f.close()

    @staticmethod
    def get_validation_accuracy(
            history: History,
            epochs: int
    ) -> float:
        y: np.array = np.array(history.history.get('val_precision'))
        x: np.array = np.arange(0, epochs)
        slope, intercept = np.polyfit(x, y, 1)
        return - 1 * slope

    @staticmethod
    def get_log_line(
            genome: str,
            total_params: int,
            history: History
    ) -> str:
        return (
            f"network: {genome}, "
            f"n_params: {total_params}, "
            f"loss: {history.history.get('val_loss')[-1]}, "
            f"accuracy: {history.history.get('val_accuracy')[-1]}, "
            f"precision: {history.history.get('val_precision')[-1]}, "
            f"recall: {history.history.get('val_recall')[-1]}, "
            f"auc: {history.history.get('val_auc')[-1]}\n"
        )

    def _evaluate(self, X, out, *args, **kwargs):
        param_score: List[int] = []
        validation_score: List[float] = []
        constraint_list: List[float] = []
        for i in range(X.shape[0]):
            genome: str = ''.join(['1' if bit else '0' for bit in X[i]])
            # print(genome)
            try:
                model: Model = translate_genome(
                    genome,
                    self.n_components_box,
                    self.n_boxes,
                    self.input_size,
                    self.n_base_filters,
                    self.n_classes
                )
                if model is None:
                    param_score.append(10000)
                    validation_score.append(10000)
                    constraint_list.append(1)
                else:
                    model.compile(
                        optimizer="adam",
                        loss="categorical_crossentropy",
                        metrics=[
                            Precision(name="precision"),
                            Recall(name="recall"),
                            "accuracy",
                            AUC(name="auc", curve="PR")
                        ]
                    )
                    history: History = model.fit(
                        x=self.training_set_generator,
                        steps_per_epoch=self.training_set_generator.samples // self.batch_size,
                        validation_data=self.validation_set_generator,
                        validation_steps=self.validation_set_generator.samples // self.batch_size,
                        epochs=self.max_epochs,
                        batch_size=self.batch_size,
                        verbose=1
                    )
                    f1 = self.get_total_params(model)
                    f2 = self.get_validation_accuracy(
                        history,
                        self.max_epochs
                    )
                    log_str = self.get_log_line(
                        genome,
                        f1,
                        history
                    )
                    print(log_str)
                    self.save_logs(log_str)
                    param_score.append(f1)
                    validation_score.append(f2)
                    constraint_list.append(-1)
            except:
                param_score.append(10000)
                validation_score.append(10000)
                constraint_list.append(1)

        out["F"] = np.column_stack([param_score, validation_score])
        out["G"] = np.column_stack([constraint_list])
