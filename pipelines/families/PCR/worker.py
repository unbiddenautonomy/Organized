"""PCR subprocess; reads normalized quote batches from the existing app."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from .runtime import PCRRuntime
from .storage import atomic_json
from .worker_lock import single_writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    context = json.loads(args.context.read_text(encoding="utf-8"))
    if args.inputs:
        context["source_mode"] = context.get("source_mode", context.get("mode", "live"))
        context["mode"] = "recorded_replay"
    with single_writer(args.output):
        consume(args, context)


def consume(args, context):
    runtime = PCRRuntime(context, args.output, config=context.get("initial_config") if args.inputs else None)
    stream = args.inputs.open(encoding="utf-8") if args.inputs else sys.stdin
    try:
        for line in stream:
            event = json.loads(line)
            now = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
            delay = (now - datetime.fromisoformat(event["timestamp"])).total_seconds()
            runtime.on_snapshot(event, processing_delay=delay)
    except Exception as error:
        runtime.fault = f"{type(error).__name__}: {error}"
        runtime.checkpoint("FAULT")
        raise
    finally:
        if args.inputs:
            stream.close()
        runtime.checkpoint("FAULT" if runtime.fault else "STOPPED")


if __name__ == "__main__":
    main()
