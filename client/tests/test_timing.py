import pytest

from recall_client.timing import ChunkClock


def test_first_chunk_is_stamped_at_its_first_sample():
    clock = ChunkClock(16000)
    assert clock.stamp(1600, now=10.0) == pytest.approx(9.9)


def test_later_chunks_are_contiguous_by_sample_count_not_wall_clock():
    clock = ChunkClock(16000)
    first = clock.stamp(1600, now=10.0)
    second = clock.stamp(1600, now=10.25)  # callback jitter must not create gaps
    third = clock.stamp(1600, now=10.27)
    assert second - first == pytest.approx(0.1)
    assert third - second == pytest.approx(0.1)


def test_resync_reanchors_after_dropped_samples():
    clock = ChunkClock(16000)
    clock.stamp(1600, now=10.0)
    clock.resync()
    assert clock.stamp(1600, now=15.0) == pytest.approx(14.9)
