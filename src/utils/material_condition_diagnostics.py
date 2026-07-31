from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaterialRecord:
    model: str
    mat_type: int
    log10_e: float
    nu: float


def build_parameter_derangement(
    records: list[MaterialRecord], seed: int
) -> dict[str, tuple[float, float]]:
    grouped: dict[int, list[MaterialRecord]] = {}
    for record in records:
        grouped.setdefault(record.mat_type, []).append(record)

    assignments: dict[str, tuple[float, float]] = {}
    for mat_type, group in grouped.items():
        ordered = sorted(group, key=lambda record: record.model)
        if len(ordered) < 2:
            raise ValueError("each material group must contain at least two records")
        rng = np.random.default_rng(seed + mat_type)
        shuffled = [ordered[index] for index in rng.permutation(len(ordered))]
        for index, record in enumerate(shuffled):
            source = shuffled[(index + 1) % len(shuffled)]
            assignments[record.model] = (source.log10_e, source.nu)
    return assignments


def rotate_material_type(mat_type: int) -> int:
    if mat_type not in (0, 1, 2):
        raise ValueError("mat_type expected one of 0, 1, 2")
    return (mat_type + 1) % 3
