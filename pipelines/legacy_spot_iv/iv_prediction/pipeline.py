"""IV pipeline with contract-level OI message integration."""

from __future__ import annotations

from ..iv_inference import IvPredictionPipeline as _BaseIvPredictionPipeline

from voltr.feed.messages.open_interest import OiDecodeError, decode_oi_message


class IvPredictionPipeline(_BaseIvPredictionPipeline):
    def on_oi_message(self, message: bytes) -> None:
        try:
            updates = decode_oi_message(message)
        except OiDecodeError:
            return
        for update in updates:
            self.on_oi_update(update.token, update.open_interest, update.log_time, update.transaction_code)
