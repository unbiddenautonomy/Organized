import threading
import time
import sys

from live.feed_fanout import FeedChannel, Fanout, read_json

from live.feed_journal import FeedJournal


def test_minute_history_and_independent_replay(tmp_path):
    journal = FeedJournal(tmp_path, retention_minutes=65)
    base = int(time.time() // 60) * 60
    try:
        for seq in range(1, 5):
            journal.append(b'F', seq, base + (seq - 1) * 60, bytes([seq]))
        journal.handle.flush()
        stopped = threading.Event()
        first = journal.reader(stopped.is_set)
        second = journal.reader(stopped.is_set)
        assert [next(first)[1] for _ in range(4)] == [1, 2, 3, 4]
        assert [next(second)[1] for _ in range(4)] == [1, 2, 3, 4]
        journal.append(b'C', 5, base + 181, b'live')
        journal.handle.flush()
        assert next(first)[1:] == (5, base + 181, b'live')
        assert next(second)[1:] == (5, base + 181, b'live')
        stopped.set()
        first.close()
        second.close()
    finally:
        journal.close()


def test_history_retains_at_least_one_hour(tmp_path):
    journal = FeedJournal(tmp_path, retention_minutes=65)
    base = 2_000_000_000 // 60 * 60
    try:
        for minute in range(70):
            journal.append(b'F', minute + 1, base + minute * 60, b'x')
        files = sorted(tmp_path.glob('*.feed'))
        assert len(files) == 65
        assert int(files[0].stem) == base // 60 + 5
    finally:
        journal.close()
    restored = FeedJournal(tmp_path)
    try:
        assert restored.recovered and restored.last_sequence == 70
        restored.append(b'F', 71, base + 70 * 60, b'y')
        assert restored.last_sequence == 71
    finally:
        restored.close()


def test_slow_worker_does_not_drop_for_healthy_worker(tmp_path):
    script = tmp_path / 'consumer.py'
    script.write_text('''import sys,time,json
from pathlib import Path
sys.path.insert(0,sys.argv[3])
from live.feed_fanout import read_frame,atomic_json
name,path=sys.argv[1],Path(sys.argv[2]);count=0
while True:
    frame=read_frame(sys.stdin.buffer)
    if frame is None:break
    if name=='slow':time.sleep(.1)
    count+=1
    if count%10==0:atomic_json(path,{'count':count,'sequence':frame[1]})
''')
    root = __import__('pathlib').Path(__file__).resolve().parent
    journal = FeedJournal(tmp_path / 'feed')
    channels = {}
    try:
        for name in ('slow', 'healthy'):
            channels[name] = FeedChannel(name, [sys.executable, '-u', str(script), name,
                str(tmp_path / (name + '.json')), str(root)], root, tmp_path / (name + '.log'),
                capacity=2, journal=journal)
        fanout = Fanout(journal)
        fanout.channels = channels
        for _ in range(100):
            fanout.publish(b'F', b'x' * 512)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and read_json(tmp_path / 'healthy.json', {}).get('count') != 100:
            time.sleep(.05)
        assert read_json(tmp_path / 'healthy.json')['sequence'] == 100
        assert channels['healthy'].dropped == channels['slow'].dropped == 0
        assert channels['slow'].delivered < 100
    finally:
        for channel in channels.values():
            channel.close()
        journal.close()
