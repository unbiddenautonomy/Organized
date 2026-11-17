"""VolTr common layer: code shared by every BANKNIFTY paper-trading pipeline.

Pipelines import from here; nothing here imports a pipeline.

    core         market clock, machine settings, crash-safe JSON files and locks
    feed         UDP receiver, LZO decoder, message decoders, order book, raw journal
    instruments  FO contract master and the daily download
    market       bars, implied volatility and greeks, the shared ATM feature stream
    service      daily supervisor, workers, feed fan-out and the pipeline registry
    web          signal and position sounds
"""
