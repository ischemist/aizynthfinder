from __future__ import annotations

import numpy as np
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

    from tensorflow.keras import regularizers
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.metrics import top_k_categorical_accuracy
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import Sequence

    class ExpansionSequence(_BaseExpansionSequence, Sequence):
        pass

    train_seq = ExpansionSequence(config, "training")
    valid_seq = ExpansionSequence(config, "validation")

    model = Sequential()
    model.add(
        Dense(
            config.hidden_nodes,
            input_shape=(train_seq.input_dim,),
            activation="elu",
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(Dropout(config.drop_out))
    model.add(Dense(train_seq.output_dim, activation="softmax"))

    top10_acc = functools.partial(top_k_categorical_accuracy, k=10)
    top10_acc.__name__ = "top10_acc"
    top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
    top50_acc.__name__ = "top50_acc"

    config.output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)
    model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss="categorical_crossentropy",
        metrics=["accuracy", "top_k_categorical_accuracy", top10_acc, top50_acc],
    )
    model.fit(
        train_seq,
        epochs=config.epochs,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10),
            CSVLogger(config.filename("_keras_training.log"), append=True),
            ModelCheckpoint(checkpoint_path / "keras_model.hdf5", monitor="loss", save_best_only=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_delta=0.000001),
        ],
        validation_data=valid_seq,
        shuffle=True,
    )
    model.save(checkpoint_path / "keras_model_final.hdf5")
