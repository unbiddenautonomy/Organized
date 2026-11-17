"""A feed gap blocks paper entries until a fresh feature window exists."""
from threading import RLock
from types import SimpleNamespace

from pipelines.ivrv.pipeline import IVRVAdapter


def test_ivrv_gap_recovery_needs_61_complete_minutes_after_latest_gap():
    adapter = IVRVAdapter.__new__(IVRVAdapter)
    adapter.engine = SimpleNamespace(
        lock=RLock(), latest_spot_minute=100, latest_future_minute=100,
        last_finalized_minute=160, rows_by_minute={m: {} for m in range(101, 161)},
    )
    adapter.note_feed_gap()
    assert not adapter.feed_gap_recovered()

    adapter.engine.last_finalized_minute = 161
    adapter.engine.rows_by_minute[161] = {}
    assert adapter.feed_gap_recovered()

    del adapter.engine.rows_by_minute[130]
    assert not adapter.feed_gap_recovered()

    adapter.engine.latest_spot_minute = 170
    adapter.engine.latest_future_minute = 170
    adapter.note_feed_gap()
    adapter.engine.last_finalized_minute = 231
    adapter.engine.rows_by_minute.update({m: {} for m in range(171, 232)})
    assert adapter.feed_gap_recovered()
