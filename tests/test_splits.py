import numpy as np
import pytest

from alpha_lab.validation.splits import PurgedKFold, purged_kfold_indices


def test_test_folds_cover_all_samples_exactly_once():
    seen = np.zeros(100, dtype=int)
    for _, test in purged_kfold_indices(100, n_splits=5, purge=2, embargo=2):
        seen[test] += 1

    assert (seen == 1).all()


def test_test_folds_cover_all_samples_exactly_once_nondivisible():
    """n_samples не делится на n_splits: тестовые фолды всё равно образуют разбиение."""
    n, k = 17, 5
    seen = np.zeros(n, dtype=int)
    folds = []
    for _, test in purged_kfold_indices(n, n_splits=k, purge=1, embargo=1):
        seen[test] += 1
        folds.append(np.asarray(test))

    assert (seen == 1).all()
    # Фолды идут подряд, без пропусков и перекрытий.
    assert np.array_equal(np.concatenate(folds), np.arange(n))


def test_train_never_overlaps_test():
    for train, test in purged_kfold_indices(200, n_splits=4, purge=10, embargo=5):
        assert len(np.intersect1d(train, test)) == 0


def test_purge_removes_neighbours_of_test():
    for train, test in purged_kfold_indices(200, n_splits=4, purge=10, embargo=0):
        lo, hi = test.min(), test.max()
        inside = train[(train >= lo - 10) & (train <= hi + 10)]
        assert len(inside) == 0


def test_embargo_removes_samples_after_test():
    for train, test in purged_kfold_indices(200, n_splits=4, purge=0, embargo=5):
        hi = test.max()
        after = train[(train > hi) & (train <= hi + 5)]
        assert len(after) == 0


def test_zero_purge_and_embargo_keeps_all_other_samples():
    n, k = 100, 5
    for train, test in purged_kfold_indices(n, n_splits=k, purge=0, embargo=0):
        assert len(train) + len(test) == n


def test_zero_purge_and_embargo_keeps_all_other_samples_nondivisible():
    n, k = 17, 5
    total = 0
    for train, test in purged_kfold_indices(n, n_splits=k, purge=0, embargo=0):
        assert len(train) + len(test) == n
        total += len(test)

    assert total == n


def test_embargo_at_series_end_does_not_overflow():
    splits = list(purged_kfold_indices(50, n_splits=5, purge=0, embargo=100))

    assert len(splits) == 5


def test_embargo_at_series_end_indices_stay_in_range():
    """Клэмп на конце серии: ни один индекс не выходит за n_samples."""
    n = 50
    for train, test in purged_kfold_indices(n, n_splits=5, purge=0, embargo=100):
        assert len(test) > 0
        assert test.min() >= 0 and test.max() < n
        if len(train) > 0:
            assert train.min() >= 0 and train.max() < n


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="n_splits"):
        list(purged_kfold_indices(100, n_splits=1, purge=0, embargo=0))
    with pytest.raises(ValueError, match="purge"):
        list(purged_kfold_indices(100, n_splits=5, purge=-1, embargo=0))
    with pytest.raises(ValueError, match="embargo"):
        list(purged_kfold_indices(100, n_splits=5, purge=0, embargo=-1))
    with pytest.raises(ValueError, match="Наблюдений"):
        list(purged_kfold_indices(3, n_splits=5, purge=0, embargo=0))


def test_purged_kfold_class_matches_function():
    kf = PurgedKFold(n_splits=5, purge=3, embargo=3)
    a = list(kf.split(120))
    b = list(purged_kfold_indices(120, 5, 3, 3))

    assert len(a) == len(b)
    for (tr1, te1), (tr2, te2) in zip(a, b):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


def test_purged_kfold_defaults():
    kf = PurgedKFold()

    assert (kf.n_splits, kf.purge, kf.embargo) == (6, 50, 20)


def _blocked_by_independent_arithmetic(n, lo, hi, purge, embargo):
    """Индексы, которых не должно быть в train, посчитанные заново, без кода сплиттера.

    purge-зона симметрична: [lo - purge, hi + purge] с клэмпом по краям серии.
    embargo-зона строго после теста: [hi + 1, hi + embargo].
    """
    blocked = set(range(max(0, lo - purge), min(n, hi + purge + 1)))
    blocked |= set(range(hi + 1, min(n, hi + embargo + 1)))
    return blocked


@pytest.mark.parametrize(
    ("n", "k", "purge", "embargo"),
    [(20, 4, 2, 3), (100, 5, 2, 2), (17, 5, 3, 4), (200, 4, 10, 5),
     (50, 5, 0, 100), (5, 5, 1, 1)],
)
def test_no_train_index_inside_purge_or_embargo_zone(n, k, purge, embargo):
    """Прямой тест на утечку: train не заходит в purge/embargo-зону теста.

    Ожидаемый train считается независимой арифметикой (множества), а не
    повторным использованием границ реализации. Проверяются оба направления:
    ни один «запрещённый» индекс не попал в train и ни один разрешённый не потерян.
    """
    for train, test in purged_kfold_indices(n, n_splits=k, purge=purge, embargo=embargo):
        lo, hi = int(test.min()), int(test.max())
        blocked = _blocked_by_independent_arithmetic(n, lo, hi, purge, embargo)
        train_list = train.tolist()

        assert not (set(train_list) & blocked)
        assert not (set(train_list) & set(test.tolist()))
        assert sorted(train_list) == sorted(set(range(n)) - blocked)

        # То же самое в терминах расстояний до тестового блока.
        for t in train_list:
            if t < lo:
                assert lo - t > purge
            elif t > hi:
                assert t - hi > purge
                assert t - hi > embargo


def test_exact_index_sets_small_case():
    """Ручная арифметика для n=20, k=4, purge=2, embargo=3.

    Фолды np.array_split: [0..4], [5..9], [10..14], [15..19].
    """
    folds = list(purged_kfold_indices(20, n_splits=4, purge=2, embargo=3))

    assert [te.tolist() for _, te in folds] == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
    ]
    assert [tr.tolist() for tr, _ in folds] == [
        list(range(8, 20)),                       # тест [0..4]: blocked [0..7]
        [0, 1, 2] + list(range(13, 20)),          # тест [5..9]: blocked [3..12]
        list(range(0, 8)) + [18, 19],             # тест [10..14]: blocked [8..17]
        list(range(0, 13)),                       # тест [15..19]: blocked [13..19]
    ]
