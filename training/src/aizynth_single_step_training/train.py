from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy import sparse
from sklearn.utils import shuffle

from .config import TrainingConfig


class _BaseExpansionSequence:
    def __init__(self, config: TrainingConfig, dataset_label: str) -> None:
        self.batch_size = config.batch_size
        self.input_matrix = sparse.load_npz(config.filename(dataset_label + "_inputs"))
        self.label_matrix = sparse.load_npz(config.filename(dataset_label + "_labels"))
        self.input_dim = self.input_matrix.shape[1]
        self.output_dim = self.label_matrix.shape[1]

    def __len__(self) -> int:
        return int(np.ceil(self.label_matrix.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        return self.input_matrix[start:end].toarray(), self.label_matrix[start:end].toarray()

    def on_epoch_end(self) -> None:
        self.input_matrix, self.label_matrix = shuffle(
            self.input_matrix, self.label_matrix, random_state=0
        )


def train_expansion_model(config: TrainingConfig) -> None:
    import functools
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import tensorflow as tf
    from tensorflow.keras import regularizers
    from tensorflow.keras.callbacks import (
        Callback,
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.metrics import top_k_categorical_accuracy
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import Sequence
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except ImportError:
        pass
    tf.get_logger().setLevel("ERROR")

    class ExpansionSequence(_BaseExpansionSequence, Sequence):
        def __init__(self, config: TrainingConfig, dataset_label: str, **kwargs) -> None:
            Sequence.__init__(self, **kwargs)
            _BaseExpansionSequence.__init__(self, config, dataset_label)

    class RichTrainingProgress(Callback):
        def __init__(self, epochs: int, steps_per_epoch: int) -> None:
            super().__init__()
            self.epochs = epochs
            self.steps_per_epoch = steps_per_epoch
            self.progress: Progress | None = None
            self.task_id = None
            self.current_epoch = 0

        def on_train_begin(self, logs=None) -> None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]training[/bold]"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[status]}"),
            )
            self.progress.start()
            self.task_id = self.progress.add_task(
                "training",
                total=self.epochs * self.steps_per_epoch,
                status="starting",
            )

        def on_epoch_begin(self, epoch: int, logs=None) -> None:
            self.current_epoch = epoch + 1
            self._update(status=f"epoch {epoch + 1}/{self.epochs}")

        def on_train_batch_end(self, batch: int, logs=None) -> None:
            logs = logs or {}
            self._advance(
                status=(
                    f"epoch {self._epoch_label()} "
                    f"loss={logs.get('loss', float('nan')):.4f} "
                    f"acc={logs.get('accuracy', float('nan')):.4f}"
                )
            )

        def on_epoch_end(self, epoch: int, logs=None) -> None:
            logs = logs or {}
            self._update(
                status=(
                    f"epoch {epoch + 1}/{self.epochs} "
                    f"loss={logs.get('loss', float('nan')):.4f} "
                    f"val_loss={logs.get('val_loss', float('nan')):.4f} "
                    f"val_acc={logs.get('val_accuracy', float('nan')):.4f} "
                    f"val_top10={logs.get('val_top10_acc', float('nan')):.4f}"
                )
            )

        def on_train_end(self, logs=None) -> None:
            self._update(status="done")
            if self.progress:
                self.progress.stop()

        def _advance(self, status: str) -> None:
            if self.progress and self.task_id is not None:
                self.progress.advance(self.task_id)
                self.progress.update(self.task_id, status=status)

        def _update(self, status: str) -> None:
            if self.progress and self.task_id is not None:
                self.progress.update(self.task_id, status=status)

        def _epoch_label(self) -> str:
            return f"{self.current_epoch}/{self.epochs}"

    train_seq = ExpansionSequence(config, "training")
    valid_seq = _load_optional_sequence(ExpansionSequence, config, "validation")
    has_validation = valid_seq is not None and valid_seq.label_matrix.shape[0] > 0

    model = Sequential(
        [
            Input(shape=(train_seq.input_dim,)),
            Dense(
                config.hidden_nodes,
                activation="elu",
                kernel_regularizer=regularizers.l2(0.001),
            ),
            Dropout(config.drop_out),
            Dense(train_seq.output_dim, activation="softmax"),
        ]
    )

    top10_acc = functools.partial(top_k_categorical_accuracy, k=10)
    top10_acc.__name__ = "top10_acc"
    top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
    top50_acc.__name__ = "top50_acc"

    config.output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)
    monitor_metric = "val_loss" if has_validation else "loss"
    best_model_path = checkpoint_path / f"keras_model_best_{monitor_metric}.keras"
    final_hdf5_path = checkpoint_path / "keras_model.hdf5"
    model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss="categorical_crossentropy",
        metrics=["accuracy", "top_k_categorical_accuracy", top10_acc, top50_acc],
        jit_compile=False,
    )
    callbacks = [
        CSVLogger(config.filename("_keras_training.log"), append=True),
        ModelCheckpoint(
            best_model_path,
            monitor=monitor_metric,
            mode="min",
            save_best_only=True,
        ),
        ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=5, min_delta=0.000001),
    ]
    if has_validation:
        callbacks.insert(0, EarlyStopping(monitor="val_loss", patience=10))
    if config.fit_verbose == 0:
        callbacks.insert(0, RichTrainingProgress(config.epochs, len(train_seq)))

    fit_kwargs = {
        "epochs": config.epochs,
        "verbose": config.fit_verbose,
        "callbacks": callbacks,
        "shuffle": True,
    }
    if has_validation:
        fit_kwargs["validation_data"] = valid_seq

    model.fit(
        train_seq,
        **fit_kwargs,
    )
    best_model = load_model(best_model_path, compile=False)
    best_model.save(final_hdf5_path)
    model.save(checkpoint_path / "keras_model_final.hdf5")


def _load_optional_sequence(sequence_cls, config: TrainingConfig, dataset_label: str):
    inputs = config.filename(dataset_label + "_inputs")
    labels = config.filename(dataset_label + "_labels")
    if not Path(inputs).exists() or not Path(labels).exists():
        return None
    return sequence_cls(config, dataset_label)
