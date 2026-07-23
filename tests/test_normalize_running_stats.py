import numpy as np
import pytest

from lingbotvla.utils.normalize import RunningStats, RunningStatsState


def _near_identity_values() -> np.ndarray:
    return np.linspace(0.9995, 1.0, 100_003, dtype=np.float32)[:, None]


def test_running_stats_preserves_small_variance_near_one_across_batches() -> None:
    values = _near_identity_values()
    expected = values.astype(np.float64).std(axis=0, ddof=0)

    stats = RunningStats()
    for batch in np.array_split(values, 37):
        stats.update(batch)

    actual = np.asarray(stats.get_statistics().std, dtype=np.float64)
    assert np.isfinite(actual).all()
    assert (actual > 0).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0.0)


def test_running_stats_single_and_multi_batch_results_match() -> None:
    values = _near_identity_values()

    single = RunningStats()
    single.update(values)

    multi = RunningStats()
    for batch in np.array_split(values, 37):
        multi.update(batch)

    np.testing.assert_allclose(
        np.asarray(multi.get_statistics().std, dtype=np.float64),
        np.asarray(single.get_statistics().std, dtype=np.float64),
        rtol=1e-7,
        atol=0.0,
    )


def test_running_stats_merge_preserves_small_variance() -> None:
    values = _near_identity_values()
    expected = values.astype(np.float64).std(axis=0, ddof=0)
    shards = []
    for batch in np.array_split(values, 7):
        shard = RunningStats()
        shard.update(batch)
        shards.append(shard)

    merged = RunningStats.merge(shards)
    actual = np.asarray(merged.get_statistics().std, dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0.0)


def test_running_stats_state_round_trip_uses_float64_accumulators() -> None:
    values = _near_identity_values()
    stats = RunningStats()
    stats.update(values)

    restored = RunningStats.from_state(RunningStatsState(**stats.get_state().model_dump()))
    assert restored._mean.dtype == np.float64
    assert restored._mean_of_squares.dtype == np.float64
    np.testing.assert_allclose(
        np.asarray(restored.get_statistics().std, dtype=np.float64),
        values.astype(np.float64).std(axis=0, ddof=0),
        rtol=1e-7,
        atol=0.0,
    )


def test_running_stats_allows_truly_constant_dimensions() -> None:
    stats = RunningStats()
    stats.update(np.ones((128, 1), dtype=np.float32))
    result = stats.get_statistics()
    np.testing.assert_array_equal(np.asarray(result.std), np.zeros(1))


def test_running_stats_rejects_zero_std_for_varying_dimension() -> None:
    stats = RunningStats()
    stats.update(np.array([[0.999], [1.0]], dtype=np.float32))
    stats._mean_of_squares = np.square(stats._mean)

    with pytest.raises(ValueError, match="non-positive std"):
        stats.get_statistics()
