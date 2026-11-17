"""PCR's file conventions: every JSON value passes through json_clean first."""
from functools import partial

from voltr.core.io import append_jsonl, atomic_write_json, json_clean as clean, read_jsonl as read_lines, read_text  # noqa: F401

atomic_json = partial(atomic_write_json, prepare=clean)
append = partial(append_jsonl, prepare=clean)
