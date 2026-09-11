"""Purged K-Fold с embargo (López de Prado, AFML гл. 7).

Зачем: признаки считаются на скользящем окне, поэтому train и test пересекаются
по времени, и модель косвенно видит будущее. Обычный KFold этого не замечает.

purge  — из train удаляются наблюдения, чьё окно признаков заглядывает в test.
embargo — дополнительно удаляются наблюдения сразу ПОСЛЕ test: признаки инерционны,
          и информация из test «протекает» в следующие за ним бары.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


def purged_kfold_indices(n_samples: int, n_splits: int, purge: int,
                         embargo: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError(f"n_splits должен быть ≥ 2, получено {n_splits}")
    if purge < 0:
        raise ValueError(f"purge не может быть отрицательным, получено {purge}")
    if embargo < 0:
        raise ValueError(f"embargo не может быть отрицательным, получено {embargo}")
    if n_samples < n_splits:
        raise ValueError(
            f"Наблюдений ({n_samples}) меньше, чем фолдов ({n_splits})"
        )

    indices = np.arange(n_samples)
    for test in np.array_split(indices, n_splits):
        test_lo, test_hi = int(test[0]), int(test[-1])
        train_mask = np.ones(n_samples, dtype=bool)

        # Сам тест исключается всегда
        train_mask[test_lo:test_hi + 1] = False
        # Purge: заглядываем на purge назад и вперёд от границ теста
        train_mask[max(0, test_lo - purge):min(n_samples, test_hi + 1 + purge)] = False
        # Embargo: зазор сразу после теста
        train_mask[test_hi + 1:min(n_samples, test_hi + 1 + embargo)] = False

        yield indices[train_mask], test


@dataclass(frozen=True)
class PurgedKFold:
    n_splits: int = 6
    purge: int = 50
    embargo: int = 20

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return purged_kfold_indices(n_samples, self.n_splits, self.purge, self.embargo)
