"""An OS lock prevents two PCR workers from trading the same ledger."""
from contextlib import ExitStack, contextmanager
from pathlib import Path

from voltr.core.io import file_lock


@contextmanager
def single_writer(directory):
    with ExitStack() as stack:
        try:
            stack.enter_context(file_lock(Path(directory) / "worker.lock"))
        except OSError as error:
            raise RuntimeError(f"Another PCR worker owns ledger {directory}") from error
        yield
