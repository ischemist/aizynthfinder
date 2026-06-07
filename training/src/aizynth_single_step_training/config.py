from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    output_path: Path
    file_prefix: str = ""
    batch_size: int = 256
    epochs: int = 100
    fingerprint_radius: int = 2
    fingerprint_len: int = 2048
    template_occurrence: int = 3
    split_size: dict[str, float] = field(
        default_factory=lambda: {"training": 0.9, "validation": 0.05, "testing": 0.05}
    )
    hidden_nodes: int = 512
    drop_out: float = 0.4
    fit_verbose: int = 0

    def filename(self, label: str) -> Path:
        postfix = {
            "library": "_template_library.csv.gz",
            "all_library": "_all.csv.gz",
            "training_labels": "_training_labels.npz",
            "validation_labels": "_validation_labels.npz",
            "testing_labels": "_testing_labels.npz",
            "training_inputs": "_training_inputs.npz",
            "validation_inputs": "_validation_inputs.npz",
            "testing_inputs": "_testing_inputs.npz",
            "training_library": "_training.csv.gz",
            "validation_library": "_validation.csv.gz",
            "testing_library": "_testing.csv.gz",
            "unique_templates_hdf5": "_unique_templates.hdf5",
            "unique_templates_csv": "_unique_templates.csv.gz",
            "_keras_training.log": "_keras_training.log",
        }.get(label, label)
        return self.output_path / f"{self.file_prefix}{postfix}"
